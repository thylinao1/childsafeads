"""M1 encoder lane trainer: one (arch, seed) run = 5 sequential GroupKFold folds.

Writes per-seed npz to <preds>/ :  <member>-s<seed>_<task>_<split>.npz
  oof: ids (2353), probs (N x K)  - each row predicted by the model that held out its fold
  dev: ids (504),  probs (N x K, mean over 5 fold models), dev_folds (5 x N x K)
Epoch is picked ONCE per run on mean-across-folds OOF macro-F1 per epoch (naive decisions),
never per-fold.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
# h100-47 nodes (xgpi*) crash in SDPA's cuDNN backend with
# CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH (2026-08-11, job 725142); the h100-96
# smoke node and every a100 run never selected it. Fall back to flash/mem-efficient.
if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
    torch.backends.cuda.enable_cudnn_sdp(False)
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from common import (ARCH_MODEL, BUDGETS, TASKS, FieldDropper, build_text, extract_fields,
                    label_arrays, load_jsonl, naive_score)
from model import MultiHeadEncoder, make_losses


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TrainDS(Dataset):
    def __init__(self, fields_list, y1, y2, y3, tok, max_len, dropper=None):
        self.fields_list = fields_list
        self.y1, self.y2, self.y3 = y1, y2, y3
        self.tok, self.max_len, self.dropper = tok, max_len, dropper

    def __len__(self):
        return len(self.fields_list)

    def __getitem__(self, i):
        dropped = self.dropper.sample() if self.dropper else ()
        enc = self.tok(build_text(self.fields_list[i], dropped), truncation=True,
                       max_length=self.max_len, padding=False)
        item = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        if self.y1 is not None:
            item["y1"] = int(self.y1[i])
            item["y2"] = self.y2[i]
            item["y3"] = self.y3[i]
        return item


def collate(tok):
    def fn(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id
        ids = torch.full((len(batch), maxlen), pad, dtype=torch.long)
        am = torch.zeros((len(batch), maxlen), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"], dtype=torch.long)
            am[i, :n] = torch.tensor(b["attention_mask"], dtype=torch.long)
        out = {"input_ids": ids, "attention_mask": am}
        if "y1" in batch[0]:
            out["y1"] = torch.tensor([b["y1"] for b in batch], dtype=torch.long)
            out["y2"] = torch.tensor(np.stack([b["y2"] for b in batch]))
            out["y3"] = torch.tensor(np.stack([b["y3"] for b in batch]))
        return out
    return fn


@torch.no_grad()
def predict(model, loader, device, use_bf16):
    # Inference always runs fp32: DeBERTa-v3 emitted non-finite logits under
    # bf16 autocast inference (smoke 723640); fp32 predict is cheap at this scale.
    del use_bf16
    model.eval()
    p1s, p2s, p3s = [], [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        am = batch["attention_mask"].to(device)
        l1, l2, l3 = model(ids, am)
        for name, t in (("st1", l1), ("st2", l2), ("st3", l3)):
            if not torch.isfinite(t).all():
                raise RuntimeError(f"non-finite {name} logits at predict "
                                   f"(bad={(~torch.isfinite(t)).sum().item()}/{t.numel()})")
        p1s.append(torch.softmax(l1.float(), -1).cpu().numpy())
        p2s.append(torch.sigmoid(l2.float()).cpu().numpy())
        p3s.append(torch.sigmoid(l3.float()).cpu().numpy())
    return (np.concatenate(p1s).astype(np.float32),
            np.concatenate(p2s).astype(np.float32),
            np.concatenate(p3s).astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["deberta", "modernbert"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--repo", default=os.path.expanduser("~/childsafeads"))
    ap.add_argument("--preds", default=None)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr-trunk", type=float, default=1e-5)
    ap.add_argument("--lr-heads", type=float, default=5e-5)
    ap.add_argument("--smoke", action="store_true", help="tiny run: 2 folds x 1 epoch x few steps")
    args = ap.parse_args()

    member = {"deberta": "m1d", "modernbert": "m1m"}[args.arch]
    preds_dir = args.preds or os.path.join(args.repo, "preds")
    os.makedirs(preds_dir, exist_ok=True)
    model_name = ARCH_MODEL[args.arch]
    budgets = BUDGETS[args.arch]
    max_len = budgets["max_len"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # The smoke-723662 divergence was fp16 WEIGHTS (transformers v5 honoring
    # deberta's config dtype), fixed in model.py by forcing fp32 weights.
    # bf16 autocast over fp32 weights is the standard stable configuration; the
    # finite-loss guard below fails fast if any instability remains.
    use_bf16 = device == "cuda"

    set_seed(args.seed)
    print(f"[run] arch={args.arch} member={member} seed={args.seed} model={model_name} "
          f"max_len={max_len} device={device} bf16={use_bf16}", flush=True)

    train_rows = load_jsonl(os.path.join(args.repo, "data/train.jsonl"))
    dev_rows = load_jsonl(os.path.join(args.repo, "data/dev.jsonl"))
    test_path = os.path.join(args.repo, "data/test.jsonl")
    test_rows = load_jsonl(test_path) if os.path.exists(test_path) else []
    folds_map = json.load(open(os.path.join(args.repo, "eval/folds.json")))
    fold_of = np.array([folds_map[r["channel_context"]["channelID"]] for r in train_rows])
    train_ids = np.array([r["instanceID"] for r in train_rows], dtype=str)
    dev_ids = np.array([r["instanceID"] for r in dev_rows], dtype=str)
    test_ids = np.array([r["instanceID"] for r in test_rows], dtype=str)
    y1, y2, y3 = label_arrays(train_rows)
    n_tr, n_dev, n_test = len(train_rows), len(dev_rows), len(test_rows)

    tok = AutoTokenizer.from_pretrained(model_name)
    t0 = time.time()
    tr_fields = [extract_fields(r, tok, budgets) for r in train_rows]
    dv_fields = [extract_fields(r, tok, budgets) for r in dev_rows]
    ts_fields = [extract_fields(r, tok, budgets) for r in test_rows]
    print(f"[fields] budget-truncated in {time.time()-t0:.0f}s", flush=True)

    coll = collate(tok)
    dev_loader = DataLoader(TrainDS(dv_fields, None, None, None, tok, max_len), batch_size=32,
                            shuffle=False, collate_fn=coll)
    test_loader = (DataLoader(TrainDS(ts_fields, None, None, None, tok, max_len), batch_size=32,
                              shuffle=False, collate_fn=coll) if test_rows else None)

    folds = [0, 1] if args.smoke else [0, 1, 2, 3, 4]
    epochs = 1 if args.smoke else args.epochs
    # storage: per epoch -> oof probs assembled over folds; per epoch per fold -> dev probs
    oof = {t: np.full((epochs, n_tr, len(TASKS[t])), np.nan, dtype=np.float32) for t in TASKS}
    devp = {t: np.zeros((epochs, 5, n_dev, len(TASKS[t])), dtype=np.float32) for t in TASKS}
    testp = {t: np.zeros((epochs, 5, n_test, len(TASKS[t])), dtype=np.float32) for t in TASKS}

    for fold in folds:
        tr_idx = np.where(fold_of != fold)[0]
        va_idx = np.where(fold_of == fold)[0]
        if args.smoke:
            tr_idx = tr_idx[:64]
        print(f"[fold {fold}] train={len(tr_idx)} val={len(va_idx)}", flush=True)
        set_seed(args.seed * 1000 + fold)

        model = MultiHeadEncoder(model_name).to(device)
        ce, bce2, bce3 = make_losses(y1[tr_idx], y2[tr_idx], y3[tr_idx], device)
        opt = torch.optim.AdamW(model.param_groups(args.lr_trunk, args.lr_heads))
        total_steps = max(1, (len(tr_idx) // (args.batch * args.accum)) * epochs)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=[g["lr"] for g in opt.param_groups], total_steps=total_steps,
            pct_start=0.1, anneal_strategy="linear")

        dropper = FieldDropper(args.seed * 100 + fold)
        tr_ds = TrainDS([tr_fields[i] for i in tr_idx], y1[tr_idx], y2[tr_idx], y3[tr_idx],
                        tok, max_len, dropper)
        g = torch.Generator(); g.manual_seed(args.seed * 10 + fold)
        tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, collate_fn=coll,
                               generator=g, drop_last=True)
        va_loader = DataLoader(TrainDS([tr_fields[i] for i in va_idx], None, None, None,
                                       tok, max_len), batch_size=32, shuffle=False,
                               collate_fn=coll)

        for ep in range(epochs):
            model.train()
            t0, running, nb = time.time(), 0.0, 0
            opt.zero_grad(set_to_none=True)
            for step, batch in enumerate(tr_loader):
                if args.smoke and step >= 4:
                    break
                ids = batch["input_ids"].to(device)
                am = batch["attention_mask"].to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
                    l1, l2, l3 = model(ids, am)
                    loss = (ce(l1.float(), batch["y1"].to(device))
                            + bce2(l2.float(), batch["y2"].to(device))
                            + bce3(l3.float(), batch["y3"].to(device)))
                (loss / args.accum).backward()
                if (step + 1) % args.accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    if sched.last_epoch < total_steps - 1:
                        sched.step()
                    opt.zero_grad(set_to_none=True)
                lv = loss.item()
                if not np.isfinite(lv):
                    raise RuntimeError(f"non-finite loss at fold {fold} ep {ep} step {step}")
                running += lv; nb += 1
            p1, p2, p3 = predict(model, va_loader, device, use_bf16)
            oof["st1"][ep, va_idx], oof["st2"][ep, va_idx], oof["st3"][ep, va_idx] = p1, p2, p3
            d1, d2, d3 = predict(model, dev_loader, device, use_bf16)
            devp["st1"][ep, fold], devp["st2"][ep, fold], devp["st3"][ep, fold] = d1, d2, d3
            if test_rows:
                s1, s2, s3 = predict(model, test_loader, device, use_bf16)
                testp["st1"][ep, fold], testp["st2"][ep, fold], testp["st3"][ep, fold] = s1, s2, s3
            sc, _ = naive_score(y1[va_idx], y2[va_idx], y3[va_idx], p1, p2, p3)
            print(f"[fold {fold}] epoch {ep} loss={running/max(nb,1):.4f} "
                  f"val_naive={sc:.4f} ({time.time()-t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()

    # ---- epoch selection: mean-across-folds OOF score per epoch (global, never per-fold)
    mask = ~np.isnan(oof["st1"][:, :, 0])  # epochs x N, rows covered (all, unless smoke)
    cov = np.where(mask[0])[0]
    ep_scores = []
    for ep in range(epochs):
        sc, parts = naive_score(y1[cov], y2[cov], y3[cov], oof["st1"][ep, cov],
                                oof["st2"][ep, cov], oof["st3"][ep, cov])
        ep_scores.append(sc)
        print(f"[epoch-pick] epoch {ep} oof_naive={sc:.4f} st1/2/3={parts}", flush=True)
    best_ep = int(np.argmax(ep_scores))
    print(f"[epoch-pick] BEST epoch={best_ep} score={ep_scores[best_ep]:.4f}", flush=True)

    tag = f"{member}-s{args.seed}"
    for t in TASKS:
        np.savez(os.path.join(preds_dir, f"{tag}_{t}_oof.npz"),
                 ids=train_ids[cov] if args.smoke else train_ids,
                 probs=oof[t][best_ep, cov] if args.smoke else oof[t][best_ep])
        dv_folds = devp[t][best_ep][folds]  # 5 x N x K (2 x N x K on smoke)
        np.savez(os.path.join(preds_dir, f"{tag}_{t}_dev.npz"),
                 ids=dev_ids, probs=dv_folds.mean(0).astype(np.float32),
                 dev_folds=dv_folds)
        if test_rows:
            ts_folds = testp[t][best_ep][folds]  # 5 x N x K (2 x N x K on smoke)
            np.savez(os.path.join(preds_dir, f"{tag}_{t}_test.npz"),
                     ids=test_ids, probs=ts_folds.mean(0).astype(np.float32),
                     test_folds=ts_folds)
    meta = {"member": member, "seed": args.seed, "arch": args.arch, "model": model_name,
            "best_epoch": best_ep, "epoch_scores": ep_scores, "epochs": epochs,
            "lr_trunk": args.lr_trunk, "lr_heads": args.lr_heads, "batch": args.batch,
            "accum": args.accum, "max_len": max_len, "smoke": args.smoke,
            "torch": torch.__version__}
    with open(os.path.join(preds_dir, f"{tag}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] wrote {tag}_<task>_<split>.npz + meta to {preds_dir}", flush=True)


if __name__ == "__main__":
    main()
