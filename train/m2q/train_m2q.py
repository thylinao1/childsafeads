"""M2 LoRA lane trainer (member m2q).

One invocation trains all 5 folds of one sub-task sequentially on one GPU,
predicts oof (held-out fold rows) + dev per fold, checkpoints per-fold preds,
then assembles the contract npzs:
  preds/m2q_<task>_oof.npz  : ids (N_train str), probs (N_train x K f32)
  preds/m2q_<task>_dev.npz  : ids (N_dev str), probs (mean over folds), dev_folds (5 x N_dev x K)

st1 = softmax; st2/st3 = per-class sigmoid. Seed 42 everywhere.
Model: Qwen/Qwen3-14B seq-cls + LoRA (r=32, alpha=64, all-linear, dropout 0.05), bf16,
max_len 2048, lr 1e-4 LoRA / 2e-5 head, 4 epochs, effective batch 8.
Fallback to Qwen/Qwen2.5-14B-Instruct if Qwen3 seq-cls unsupported.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
# Hopper nodes (h100-*, h200-*) crash in SDPA's cuDNN backend with
# CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH (proven on m1m jobs 725142/725144,
# 2026-08-11); force flash/mem-efficient kernels, the a100-proven path.
if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
    torch.backends.cuda.enable_cudnn_sdp(False)
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LABELS, MAX_LEN, SEED, build_input, load_data, load_test, targets

PRIMARY_MODEL = "Qwen/Qwen3-14B"
FALLBACK_MODEL = "Qwen/Qwen2.5-14B-Instruct"
LR_LORA = 1e-4
LR_HEAD = 2e-5
EPOCHS = 4
MICRO_BS = 2
ACCUM = 4  # effective batch 8
EVAL_BS = 4


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TextDS(Dataset):
    def __init__(self, rows, task, tok, with_labels=True):
        self.texts = [build_input(r) for r in rows]
        self.task = task
        self.tok = tok
        self.with_labels = with_labels
        self.ys = [targets(r, task) for r in rows] if with_labels else None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return i

    def collate(self, idxs):
        enc = self.tok([self.texts[i] for i in idxs], truncation=True, max_length=MAX_LEN,
                       padding=True, return_tensors="pt")
        out = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        if self.with_labels:
            ys = [self.ys[i] for i in idxs]
            if self.task == "st1":
                out["labels"] = torch.tensor(ys, dtype=torch.long)
            else:
                out["labels"] = torch.tensor(ys, dtype=torch.float32)
        return out


def build_model(num_labels, task):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    problem_type = "single_label_classification" if task == "st1" else "multi_label_classification"

    def _load(name):
        kw = dict(num_labels=num_labels, problem_type=problem_type, attn_implementation="sdpa")
        try:
            return AutoModelForSequenceClassification.from_pretrained(
                name, dtype=torch.bfloat16, **kw)
        except TypeError:  # transformers < 5 uses torch_dtype
            return AutoModelForSequenceClassification.from_pretrained(
                name, torch_dtype=torch.bfloat16, **kw)

    model_name = os.environ.get("M2Q_MODEL", PRIMARY_MODEL)
    for name in [model_name, FALLBACK_MODEL]:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            model = _load(name)
            print(f"[model] loaded {name}", flush=True)
            break
        except Exception as e:
            print(f"[model] FAILED to load {name}: {type(e).__name__}: {e}", flush=True)
            if name == FALLBACK_MODEL:
                raise
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.config.pad_token_id = tok.pad_token_id
    lcfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
                      target_modules="all-linear", task_type="SEQ_CLS",
                      modules_to_save=["score"])
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    return tok, model


def st1_class_weights(rows):
    import collections
    cnt = collections.Counter(r["labels"]["st1"] for r in rows)
    w = np.array([1.0 / math.sqrt(max(cnt.get(l, 0), 1)) for l in LABELS["st1"]], dtype=np.float32)
    w = w / w.mean()
    return torch.tensor(w)


@torch.no_grad()
def predict(model, ds, task, device):
    model.eval()
    dl = DataLoader(ds, batch_size=EVAL_BS, shuffle=False, collate_fn=ds.collate)
    outs = []
    for batch in dl:
        logits = model(input_ids=batch["input_ids"].to(device),
                       attention_mask=batch["attention_mask"].to(device)).logits.float()
        if task == "st1":
            outs.append(F.softmax(logits, dim=-1).cpu().numpy())
        else:
            outs.append(torch.sigmoid(logits).cpu().numpy())
    model.train()
    return np.concatenate(outs).astype(np.float32)


def quick_macro_f1(probs, rows, task):
    from sklearn.metrics import f1_score
    K = len(LABELS[task])
    if task == "st1":
        pred = probs.argmax(1)
        gold = np.array([targets(r, task) for r in rows])
        present = sorted(set(gold.tolist()) | set(pred.tolist()))
        return f1_score(gold, pred, labels=present, average="macro", zero_division=0)
    pred = (probs >= 0.5).astype(int)
    gold = np.array([targets(r, task) for r in rows], dtype=int)
    keep = [k for k in range(K) if gold[:, k].sum() > 0 or pred[:, k].sum() > 0]
    return f1_score(gold[:, keep], pred[:, keep], average="macro", zero_division=0)


def train_fold(task, fold, train_rows, dev_rows, workdir, device, test_rows=None):
    K = len(LABELS[task])
    tr = [r for r in train_rows if r["fold"] != fold]
    va = [r for r in train_rows if r["fold"] == fold]
    print(f"[{task} fold {fold}] train {len(tr)} / oof {len(va)}", flush=True)
    set_seed(SEED + fold)
    tok, model = build_model(K, task)
    model.to(device)

    ds = TextDS(tr, task, tok)
    dl = DataLoader(ds, batch_size=MICRO_BS, shuffle=True, collate_fn=ds.collate,
                    generator=torch.Generator().manual_seed(SEED + fold), drop_last=False)

    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    opt = torch.optim.AdamW([{"params": lora_params, "lr": LR_LORA},
                             {"params": head_params, "lr": LR_HEAD}], weight_decay=0.0)
    steps_per_epoch = math.ceil(len(dl) / ACCUM)
    total_steps = steps_per_epoch * EPOCHS
    from transformers import get_cosine_schedule_with_warmup
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * total_steps), total_steps)

    cw = st1_class_weights(tr).to(device) if task == "st1" else None
    model.train()
    step = 0
    t0 = time.time()
    for epoch in range(EPOCHS):
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(dl):
            logits = model(input_ids=batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device)).logits.float()
            if task == "st1":
                loss = F.cross_entropy(logits, batch["labels"].to(device), weight=cw)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, batch["labels"].to(device))
            (loss / ACCUM).backward()
            if (i + 1) % ACCUM == 0 or i == len(dl) - 1:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 25 == 0:
                    el = time.time() - t0
                    print(f"[{task} fold {fold}] ep{epoch} step {step}/{total_steps} "
                          f"loss {loss.item():.4f} elapsed {el/60:.1f}m", flush=True)
        print(f"[{task} fold {fold}] epoch {epoch} done", flush=True)

    oof_probs = predict(model, TextDS(va, task, tok, with_labels=False), task, device)
    dev_probs = predict(model, TextDS(dev_rows, task, tok, with_labels=False), task, device)
    f1 = quick_macro_f1(oof_probs, va, task)
    f1d = quick_macro_f1(dev_probs, dev_rows, task)
    print(f"[{task} fold {fold}] oof macroF1(present)={f1:.4f} dev macroF1(present)={f1d:.4f}", flush=True)
    extra = {}
    if test_rows is not None:
        extra["test_probs"] = predict(model, TextDS(test_rows, task, tok, with_labels=False), task, device)
    np.savez(os.path.join(workdir, f"fold{fold}.npz"),
             oof_ids=np.array([r["instanceID"] for r in va]),
             oof_probs=oof_probs, dev_probs=dev_probs,
             oof_f1=f1, dev_f1=f1d, seed=SEED + fold, **extra)
    model.save_pretrained(os.path.join(workdir, f"fold{fold}_adapter"))
    del model
    torch.cuda.empty_cache()


def assemble(task, train_rows, dev_rows, workdir, preds_dir, test_rows=None):
    K = len(LABELS[task])
    id2i = {r["instanceID"]: i for i, r in enumerate(train_rows)}
    oof = np.full((len(train_rows), K), np.nan, dtype=np.float32)
    dev_folds = np.zeros((5, len(dev_rows), K), dtype=np.float32)
    test_folds = np.zeros((5, len(test_rows), K), dtype=np.float32) if test_rows is not None else None
    missing_test = []
    for fold in range(5):
        z = np.load(os.path.join(workdir, f"fold{fold}.npz"), allow_pickle=True)
        for i, iid in enumerate(z["oof_ids"]):
            oof[id2i[str(iid)]] = z["oof_probs"][i]
        dev_folds[fold] = z["dev_probs"]
        if test_rows is not None:
            if "test_probs" in z.files:
                test_folds[fold] = z["test_probs"]
            else:
                missing_test.append(fold)
    assert not np.isnan(oof).any(), "OOF has unfilled rows"
    os.makedirs(preds_dir, exist_ok=True)
    np.savez(os.path.join(preds_dir, f"m2q_{task}_oof.npz"),
             ids=np.array([r["instanceID"] for r in train_rows]), probs=oof)
    np.savez(os.path.join(preds_dir, f"m2q_{task}_dev.npz"),
             ids=np.array([r["instanceID"] for r in dev_rows]),
             probs=dev_folds.mean(0).astype(np.float32), dev_folds=dev_folds)
    print(f"[{task}] assembled -> {preds_dir}/m2q_{task}_{{oof,dev}}.npz", flush=True)
    if test_rows is not None:
        assert not missing_test, (
            f"[{task}] fold npz missing test_probs for folds {missing_test}: "
            "these folds were trained before test.jsonl existed; archive work/ and rerun them")
        np.savez(os.path.join(preds_dir, f"m2q_{task}_test.npz"),
                 ids=np.array([r["instanceID"] for r in test_rows]),
                 probs=test_folds.mean(0).astype(np.float32))
        print(f"[{task}] assembled -> {preds_dir}/m2q_{task}_test.npz", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["st1", "st2", "st3"])
    ap.add_argument("--repo", default=os.path.expanduser("~/childsafeads"))
    args = ap.parse_args()

    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    train_rows, dev_rows = load_data(args.repo)
    test_rows = load_test(args.repo)
    if test_rows is not None:
        print(f"[cfg] test.jsonl present: {len(test_rows)} rows", flush=True)
    workdir = os.path.join(args.repo, "train", "m2q", "work", args.task)
    os.makedirs(workdir, exist_ok=True)
    preds_dir = os.path.join(args.repo, "preds")

    print(f"[cfg] task={args.task} seed={SEED} model={os.environ.get('M2Q_MODEL', PRIMARY_MODEL)} "
          f"lora r=32 a=64 dropout=0.05 all-linear lr={LR_LORA}/{LR_HEAD} epochs={EPOCHS} "
          f"eff_batch={MICRO_BS * ACCUM} max_len={MAX_LEN}", flush=True)
    for fold in range(5):
        done = os.path.join(workdir, f"fold{fold}.npz")
        if os.path.exists(done):
            print(f"[{args.task} fold {fold}] checkpoint exists, skipping", flush=True)
            continue
        train_fold(args.task, fold, train_rows, dev_rows, workdir, device, test_rows)
    assemble(args.task, train_rows, dev_rows, workdir, preds_dir, test_rows)


if __name__ == "__main__":
    main()
