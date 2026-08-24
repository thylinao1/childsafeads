"""The present-label-set divisor is a random variable, and it dominates the noise floor.

Under present-label-set macro-F1 a class is admitted to a sub-task's average whenever it
appears in the gold OR in the predictions of the rows being scored. The divisor of the
average is therefore a property of the sample, not of the label space. When a class has
almost no support, whether it is drawn at all flips the divisor, and the reported score
moves by a fixed multiplicative factor with no change whatsoever to the predictions.

This script measures that on the out-of-fold split. ST1 `other` has two gold instances in
2353 rows, spread over two of the 632 channels, and the system never predicts it. Channels
are resampled with replacement; the identical emitted decisions are scored in every
replicate; replicates are then partitioned by whether the draw happened to contain a gold
`other` row.

Expected, if the effect is purely the divisor: a replicate with no gold `other` averages
ST1 over 4 classes rather than 5, so its ST1 is exactly 5/4 of the other regime's.

Usage: python3 clinic/presence_artefact.py [--reps 4000]
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.join(ROOT, "decide"))
sys.path.insert(0, os.path.join(ROOT, "clinic"))

from local_scorer import ST1  # noqa: E402
import decision_layer as DL   # noqa: E402
import level_ablation as LA   # noqa: E402

RARE_CLASS = "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    params = json.load(open(os.path.join(ROOT, "decide", "params.json")))
    rows = DL.emit_preds(params["members"], "oof", params, params.get("weights"))
    preds = {r["instanceID"]: r for r in rows}
    gold = {json.loads(l)["instanceID"]: json.loads(l)
            for l in open(os.path.join(ROOT, "data", "train.jsonl")) if l.strip()}

    ids = sorted(gold)
    blocks = LA._indicator_blocks(gold, preds, ids)
    channels, agg = LA._channel_agg(blocks, ids, gold)
    nC = len(channels)

    j = ST1.index(RARE_CLASS)
    TP, FP, FN = agg["st1"]
    gold_rare_per_channel = TP[:, j] + FN[:, j]
    pred_rare = int((TP[:, j] + FP[:, j]).sum())

    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, nC, size=(args.reps, nC))
    counts = np.zeros((args.reps, nC))
    for r in range(args.reps):
        counts[r] = np.bincount(draws[r], minlength=nC)

    cols = {t: LA._macro_from_counts(*[counts @ M for M in agg[t]])
            for t in ("st1", "st2", "st3")}
    metric = sum(cols.values()) / 3
    present = (counts @ gold_rare_per_channel) > 0

    out = {
        "rare_class": RARE_CLASS,
        "gold_rare_rows": int(gold_rare_per_channel.sum()),
        "channels_with_rare": int((gold_rare_per_channel > 0).sum()),
        "channels": nC,
        "times_predicted_by_system": pred_rare,
        "p_rare_in_resample": float(present.mean()),
        "regimes": {},
    }
    for name, m in (("all", np.ones(args.reps, bool)),
                    ("rare_present", present), ("rare_absent", ~present)):
        out["regimes"][name] = {
            "reps": int(m.sum()),
            "st1_mean": float(cols["st1"][m].mean()), "st1_sd": float(cols["st1"][m].std(ddof=1)),
            "metric_mean": float(metric[m].mean()), "metric_sd": float(metric[m].std(ddof=1)),
        }
    a, b = out["regimes"]["rare_present"], out["regimes"]["rare_absent"]
    out["st1_gap_measured"] = b["st1_mean"] - a["st1_mean"]
    out["st1_gap_predicted_5_over_4"] = a["st1_mean"] * (5 / 4) - a["st1_mean"]
    out["metric_gap"] = b["metric_mean"] - a["metric_mean"]

    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(ROOT, "clinic", "presence_artefact.json"), "w"), indent=1)
    print("\nwrote clinic/presence_artefact.json")
    print(f"\nThe system predicts '{RARE_CLASS}' {pred_rare} times. Its {out['gold_rare_rows']} gold "
          f"rows move the reported metric by {out['metric_gap']:+.4f} depending only on whether they "
          f"are in the sample.")


if __name__ == "__main__":
    main()
