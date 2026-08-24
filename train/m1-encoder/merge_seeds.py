"""Average available per-seed npz into member-level files.

m1d-s42_st1_oof.npz + m1d-s43_... + m1d-s44_...  ->  m1d_st1_oof.npz
Safe to run any time; averages whatever seeds exist (asserts identical id order).
"""
import glob
import os
import re
import sys

import numpy as np

PREDS = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/childsafeads/preds")

for member in ["m1d", "m1m"]:
    for task in ["st1", "st2", "st3"]:
        for split in ["oof", "dev", "test"]:
            paths = sorted(glob.glob(os.path.join(PREDS, f"{member}-s*_{task}_{split}.npz")))
            paths = [p for p in paths if re.search(rf"{member}-s\d+_{task}_{split}\.npz$", p)]
            if not paths:
                continue
            datas = [np.load(p, allow_pickle=True) for p in paths]
            ids = datas[0]["ids"]
            assert all(np.array_equal(d["ids"], ids) for d in datas), f"id mismatch: {paths}"
            probs = np.mean([d["probs"] for d in datas], axis=0).astype(np.float32)
            out = {"ids": ids, "probs": probs}
            if split == "dev" and all("dev_folds" in d for d in datas):
                out["dev_folds"] = np.mean([d["dev_folds"] for d in datas], axis=0).astype(np.float32)
            outp = os.path.join(PREDS, f"{member}_{task}_{split}.npz")
            np.savez(outp, **out)
            seeds = [re.search(r"-s(\d+)_", os.path.basename(p)).group(1) for p in paths]
            print(f"[merge] {os.path.basename(outp)} <- seeds {seeds} shape={probs.shape}")
