"""Assert the npz output contract for the m3q32 lane.

Usage: python verify_npz.py <preds_dir> <member_prefix e.g. m3q32 or m3q32smoke> \
       <n_oof> <n_dev> [--allow-degenerate]
   or: python verify_npz.py <preds_dir> <member_prefix> <split>:<count> ... \
       (e.g. test:503) to check an explicit set of splits
Checks per (variant, task, split): file exists, ids unique + right count, probs
float32 N x K in [0,1], st1 rows sum to ~1, and probabilities are non-degenerate
(some value differs from the fallback constants, i.e. logprob extraction fired).
"""
import sys

import numpy as np

K = {"st1": 5, "st2": 12, "st3": 8}
FALLBACKS = {0.98, 0.02}


def main():
    preds_dir, member = sys.argv[1], sys.argv[2]
    allow_degenerate = "--allow-degenerate" in sys.argv
    rest = [a for a in sys.argv[3:] if a != "--allow-degenerate"]
    if any(":" in a for a in rest):
        n = {a.split(":")[0]: int(a.split(":")[1]) for a in rest}
    else:
        n = {"oof": int(rest[0]), "dev": int(rest[1])}
    problems = []
    for variant in ["A", "B"]:
        for task in ["st1", "st2", "st3"]:
            for split in n:
                path = f"{preds_dir}/{member}{variant}_{task}_{split}.npz"
                d = np.load(path, allow_pickle=True)
                ids, probs = d["ids"], d["probs"]
                assert len(ids) == n[split], f"{path}: {len(ids)} ids != {n[split]}"
                assert len(set(ids.tolist())) == len(ids), f"{path}: duplicate ids"
                assert probs.shape == (n[split], K[task]), f"{path}: shape {probs.shape}"
                assert probs.dtype == np.float32, f"{path}: dtype {probs.dtype}"
                assert np.all(probs >= 0) and np.all(probs <= 1), f"{path}: out of [0,1]"
                assert not np.any(np.isnan(probs)), f"{path}: NaN"
                if task == "st1":
                    s = probs.sum(axis=1)
                    assert np.allclose(s, 1.0, atol=1e-4), f"{path}: st1 rows not normalized"
                    assert "probs_raw" in d, f"{path}: missing probs_raw"
                    raw = d["probs_raw"]
                else:
                    raw = probs
                frac_fb = float(np.mean(np.isin(np.round(raw, 6), [0.98, 0.02])))
                degen = frac_fb > 0.99 or float(raw.std()) < 1e-6
                print(f"OK {path} shape={probs.shape} std={probs.std():.4f} "
                      f"fallback_frac={frac_fb:.3f}")
                if degen and not allow_degenerate:
                    problems.append(f"{path}: degenerate (fallback_frac={frac_fb:.3f}, "
                                    f"std={raw.std():.6f}) - logprob extraction not firing")
    if problems:
        print("DEGENERATE:\n" + "\n".join(problems))
        sys.exit(1)
    print("CONTRACT VERIFIED")


if __name__ == "__main__":
    main()
