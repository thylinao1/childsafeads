"""m2q test inference from saved fold adapters (no retraining).

For --task, loads the base seq-cls model fresh per fold (no adapter state bleed),
attaches work/<task>/fold{f}_adapter (LoRA + score head via modules_to_save),
predicts the test split, and writes preds/m2q_<task>_test.npz
(ids = test.jsonl order, probs = mean over folds, float32).

Usage (one GPU per task, ~60-75 min each):
  python infer_test_from_adapters.py --task st3 [--repo ~/childsafeads]
      [--folds 0,1,2,3,4] [--limit 0] [--out-suffix ""]
Smoke form (single fold, capped rows, separate output):
  python infer_test_from_adapters.py --task st3 --folds 0 --limit 8 \
      --repo ~/childsafeads_smoketest --out-suffix smoke
"""
import argparse
import os
import sys

import numpy as np
import torch
# Hopper nodes (h100-*, h200-*) crash in SDPA's cuDNN backend with
# CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH (proven on m1m jobs 725142/725144,
# 2026-08-11); force flash/mem-efficient kernels, the a100-proven path.
if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
    torch.backends.cuda.enable_cudnn_sdp(False)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LABELS, load_test  # noqa: E402
from train_m2q import PRIMARY_MODEL, FALLBACK_MODEL, TextDS, predict, set_seed  # noqa: E402
from common import SEED  # noqa: E402


def load_base(num_labels, task):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
    return tok, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["st1", "st2", "st3"])
    ap.add_argument("--repo", default=os.path.expanduser("~/childsafeads"))
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()
    set_seed(SEED)
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True

    test_rows = load_test(args.repo)
    assert test_rows, f"no data/test.jsonl under {args.repo}"
    if args.limit:
        test_rows = test_rows[:args.limit]
    K = len(LABELS[args.task])
    folds = [int(f) for f in args.folds.split(",")]
    workdir = os.path.join(args.repo, "train", "m2q", "work", args.task)
    print(f"[cfg] task={args.task} folds={folds} n_test={len(test_rows)} K={K}", flush=True)

    from peft import PeftModel
    test_folds = np.zeros((len(folds), len(test_rows), K), dtype=np.float32)
    for i, fold in enumerate(folds):
        adapter = os.path.join(workdir, f"fold{fold}_adapter")
        assert os.path.isdir(adapter), f"missing adapter {adapter}"
        tok, base = load_base(K, args.task)
        model = PeftModel.from_pretrained(base, adapter).to(device)
        model.eval()
        test_folds[i] = predict(model, TextDS(test_rows, args.task, tok, with_labels=False),
                                args.task, device)
        print(f"[{args.task} fold {fold}] test predicted "
              f"(mean={test_folds[i].mean():.4f} std={test_folds[i].std():.4f})", flush=True)
        del model, base
        torch.cuda.empty_cache()

    probs = test_folds.mean(0).astype(np.float32)
    assert np.isfinite(probs).all() and probs.std() > 1e-4, "degenerate test probs"
    suffix = f"_{args.out_suffix}" if args.out_suffix else ""
    out = os.path.join(args.repo, "preds", f"m2q_{args.task}_test{suffix}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, ids=np.array([r["instanceID"] for r in test_rows]), probs=probs)
    print(f"wrote {out} shape={probs.shape}", flush=True)


if __name__ == "__main__":
    main()
