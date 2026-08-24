"""Leave-one-out member ablation on the SIX-member shipped blend.

An earlier version of this ablation was run on the five-member blend of 2026-08-11 and
its raw output was not kept. This reruns it on the configuration that actually shipped
and writes clinic/ablate6.json.

Each arm is refit exactly as it would ship. Delta reported is (blend without member)
minus (full blend), so a NEGATIVE delta is the cost of dropping the member.
"""
import json, os, sys, shutil
import numpy as np
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "clinic")); sys.path.insert(0, os.path.join(ROOT, "eval")); sys.path.insert(0, os.path.join(ROOT, "decide"))
from oof_noise import gold_maps, pred_maps, score_subset  # noqa
import decision_layer as dl  # noqa

FULL = ["m1d", "m1m", "m3q32A", "m0s", "s3s", "m2qS2"]
N_BOOT, SEED = 2000, 20260811
P = os.path.join(ROOT, "decide", "params.json")
BK = P + ".ablate6.bak"


def emit(members):
    sys.argv = ["decision_layer.py", "fit", "--members", ",".join(members)]
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        dl.main()
    p = json.load(open(P))
    return dl.emit_preds(p["members"], "oof", p, p.get("weights"))


def main():
    shutil.copy(P, BK)
    try:
        g1, g2, g3, chan = gold_maps(os.path.join(ROOT, "data", "train.jsonl"))
        golds = (g1, g2, g3)
        by = {}
        for i in g1:
            by.setdefault(chan[i], []).append(i)
        chans = list(by)
        base = pred_maps(emit(FULL))
        out = {}
        for m in FULL:
            rest = [x for x in FULL if x != m]
            arm = pred_maps(emit(rest))
            rng = np.random.default_rng(SEED)
            reps = []
            for _ in range(N_BOOT):
                d = rng.choice(len(chans), size=len(chans), replace=True)
                ids = [i for k in d for i in by[chans[k]]]
                reps.append(np.array(score_subset(ids, golds, arm)) -
                            np.array(score_subset(ids, golds, base)))
            reps = np.array(reps)
            rec = {}
            for c, lab in enumerate(("st1", "st2", "st3", "mean")):
                col = reps[:, c]
                lo, hi = np.percentile(col, [2.5, 97.5])
                rec[lab] = dict(delta=float(col.mean()), se=float(col.std(ddof=1)),
                                ci95=[float(lo), float(hi)],
                                p_better=float((col > 0).mean()))
            out[m] = rec
            print(f"DROP {m:8s} mean {rec['mean']['delta']:+.4f} "
                  f"[{rec['mean']['ci95'][0]:+.4f},{rec['mean']['ci95'][1]:+.4f}] "
                  f"se={rec['mean']['se']:.4f} P={rec['mean']['p_better']:.3f}  | "
                  f"st1 {rec['st1']['delta']:+.4f} st2 {rec['st2']['delta']:+.4f} st3 {rec['st3']['delta']:+.4f}")
        json.dump({"full": FULL, "reps": N_BOOT, "seed": SEED, "drop": out},
                  open(os.path.join(ROOT, "clinic", "ablate6.json"), "w"), indent=1)
    finally:
        shutil.copy(BK, P); os.remove(BK)
        print("[ablate6] params.json restored")


if __name__ == "__main__":
    main()
