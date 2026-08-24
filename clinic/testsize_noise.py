"""Standard error of a 503-row (153-channel) macro-F1 estimate.

Fills the reproducibility hole: RESULTS.md quotes 0.0337 with no committed code.
Two designs are reported so the paper can state which one it uses.

  A. with-replacement channel bootstrap, 153 channels per draw   <- recommended
  B. without-replacement channel-disjoint subsample to 503 rows  <- what was
     originally run; carries a finite-population deflation of
     sqrt(1 - 153/632) = 0.871 relative to A.

Decisions are emitted ONCE, before resampling, exactly as clinic/oof_noise.py does,
so this measures sampling noise of a fixed estimate, not tuning noise.

  python clinic/testsize_noise.py                 # canonical config
  python clinic/testsize_noise.py A.jsonl B.jsonl # paired difference
"""
import json, os, sys
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "clinic"))
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.join(ROOT, "decide"))
from oof_noise import gold_maps, pred_maps, score_subset  # noqa: E402
import decision_layer as dl  # noqa: E402

N_DRAW = 2000
TEST_ROWS = 503
TEST_CHANS = 153
SEED = 20260822


def load_rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main():
    g1, g2, g3, chan = gold_maps(os.path.join(ROOT, "data", "train.jsonl"))
    golds = (g1, g2, g3)
    args = sys.argv[1:]
    if args:
        pred_sets = [pred_maps(load_rows(p)) for p in args]
        names = args
    else:
        params = json.load(open(os.path.join(ROOT, "decide", "params.json")))
        rows = dl.emit_preds(params["members"], "oof", params, params.get("weights"))
        pred_sets = [pred_maps(rows)]
        names = ["canonical"]

    by = {}
    for i in g1:
        by.setdefault(chan[i], []).append(i)
    chans = list(by)

    for design in ("A_with_replacement_153ch", "B_wor_subsample_503rows"):
        rng = np.random.default_rng(SEED)
        reps = [[] for _ in pred_sets]
        for _ in range(N_DRAW):
            if design.startswith("A"):
                draw = rng.choice(len(chans), size=TEST_CHANS, replace=True)
                ids = [i for k in draw for i in by[chans[k]]]
            else:
                order = rng.permutation(len(chans))
                ids, tot = [], 0
                for k in order:
                    ids += by[chans[k]]
                    tot += len(by[chans[k]])
                    if tot >= TEST_ROWS:
                        break
                ids = ids[:TEST_ROWS]
            for s, preds in enumerate(pred_sets):
                reps[s].append(score_subset(ids, golds, preds))
        print(f"\n=== design {design}  draws={N_DRAW} ===")
        arr = [np.array(r) for r in reps]
        for s, nm in enumerate(names):
            print(f" {nm}")
            for c, lab in enumerate(("st1", "st2", "st3", "mean")):
                col = arr[s][:, c]
                lo, hi = np.percentile(col, [2.5, 97.5])
                print(f"   {lab:5s} mean={col.mean():.4f} std.err={col.std(ddof=1):.4f} "
                      f"95% CI=[{lo:.4f}, {hi:.4f}]")
        if len(arr) == 2:
            d = arr[1] - arr[0]
            print(" paired delta (B - A)")
            for c, lab in enumerate(("st1", "st2", "st3", "mean")):
                col = d[:, c]
                lo, hi = np.percentile(col, [2.5, 97.5])
                print(f"   {lab:5s} delta={col.mean():+.4f} std.err={col.std(ddof=1):.4f} "
                      f"95% CI=[{lo:+.4f}, {hi:+.4f}]  P(better)={(col>0).mean():.3f}")


if __name__ == "__main__":
    main()
