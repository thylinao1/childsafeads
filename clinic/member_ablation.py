"""Leave-one-out member ablation for the shipped six-member blend, with paired CIs.

For each member: refit the whole decision layer without it, emit out-of-fold, and compare
against the full blend on the identical channel resample. A NEGATIVE delta (full minus
reduced, reported as the cost of removing the member) means the member is contributing.

Two details that a naive ablation gets wrong and this one handles:
  - `ST3_COL_EXTRA` and `ST3_COL_ONLY` name members explicitly and load them independently
    of the blend list, so dropping a member from `members` alone would leave it still feeding
    the ST3 columns. The dropped member is removed from those maps too.
  - Thresholds are refitted for every arm, so each reduced blend is scored with parameters
    calibrated to its own probabilities rather than to the full blend's.

Usage: python3 clinic/member_ablation.py [--reps 4000]
"""
import argparse
import copy
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.join(ROOT, "decide"))
sys.path.insert(0, os.path.join(ROOT, "clinic"))

from local_scorer import ST1, ST2, ST3  # noqa: E402
import decision_layer as DL             # noqa: E402
import level_ablation as LA             # noqa: E402

TRAIN = os.path.join(ROOT, "data", "train.jsonl")


def fit_and_emit(members, drop=None):
    """Refit ST1 tau, ST2/ST3 thresholds and the ST3 cascade for this member set, then emit OOF."""
    extra_bk, only_bk = copy.deepcopy(DL.ST3_COL_EXTRA), copy.deepcopy(DL.ST3_COL_ONLY)
    try:
        if drop:
            for m in (DL.ST3_COL_EXTRA, DL.ST3_COL_ONLY):
                for cls in list(m):
                    m[cls] = [x for x in m[cls] if x != drop]
                    if not m[cls]:
                        del m[cls]
        folds = json.load(open(os.path.join(ROOT, "eval", "folds.json")))
        rows = [json.loads(l) for l in open(TRAIN) if l.strip()]
        fold_of = np.array([folds[r["channel_context"]["channelID"]] for r in rows])

        ids1, p1 = DL.load_blend(members, "st1", "oof")
        g1 = DL.gold_sets(TRAIN, "st1")
        prior = np.array([max(sum(1 for i in ids1 if c in g1[i]), 0.5) for c in ST1]) / len(ids1)
        y1 = np.array([ST1.index(next(iter(g1[i]))) for i in ids1])
        best = (0.0, -1.0)
        for tau in np.round(np.arange(0.0, 1.51, 0.1), 2):
            dec = DL.decide_st1(p1, tau, prior)
            labs = sorted(set(y1.tolist()) | {ST1.index(d) for d in dec})
            f1s = []
            for c in labs:
                yy = y1 == c
                pp = np.array([ST1.index(d) == c for d in dec])
                tp = int((yy & pp).sum()); fp = int((~yy & pp).sum()); fn = int((yy & ~pp).sum())
                if tp or fp or fn:
                    f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
            m = sum(f1s) / len(f1s)
            if m > best[1]:
                best = (float(tau), m)

        ids2, p2 = DL.load_blend(members, "st2", "oof")
        params2 = DL.fit_thresholds(ST2, ids2, p2, DL.gold_sets(TRAIN, "st2"), fold_of)
        ids3, p3 = DL.load_blend(members, "st3", "oof")
        p3 = DL.mix_col_extra(ids3, p3, len(members), "oof")
        p3 = DL.set_col_only(ids3, p3, "oof")
        p3 = DL.apply_hard_rules_st3(ids3, p3, "oof")
        gold3 = DL.gold_sets(TRAIN, "st3")
        params3 = DL.fit_thresholds(ST3, ids3, p3, gold3, fold_of)
        nf, _ = DL.fit_st3_nf(ids3, p3, params3, gold3)
        params = {"members": members, "weights": None, "st1_tau": best[0],
                  "st1_other_tau": 1.01, "st1_prior": prior.tolist(),
                  "st2": params2, "st3": params3, "st3_nf": nf}
        emitted = DL.emit_preds(members, "oof", params, None)
        return {r["instanceID"]: r for r in emitted}
    finally:
        DL.ST3_COL_EXTRA.clear(); DL.ST3_COL_EXTRA.update(extra_bk)
        DL.ST3_COL_ONLY.clear();  DL.ST3_COL_ONLY.update(only_bk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4000)
    args = ap.parse_args()
    members = json.load(open(os.path.join(ROOT, "decide", "params.json")))["members"]
    gold = {json.loads(l)["instanceID"]: json.loads(l) for l in open(TRAIN) if l.strip()}

    print(f"full blend: {members}")
    full = fit_and_emit(members)
    out = {}
    for m in members:
        rest = [x for x in members if x != m]
        red = fit_and_emit(rest, drop=m)
        b = LA.paired_bootstrap(gold, red, full, args.reps)   # reduced minus full
        out[m] = b
        mm = b["mean"]
        print(f"\n  drop {m:<8} cost of removal on the mean {mm['delta']:+.4f}  "
              f"95% CI [{mm['ci95'][0]:+.4f}, {mm['ci95'][1]:+.4f}]  se {mm['se']:.4f}")
        for t in ("st1", "st2", "st3"):
            c = b[t]
            star = "*" if (c["ci95"][0] > 0) == (c["ci95"][1] > 0) else " "
            print(f"      {t}: {c['delta']:+.4f} CI [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] {star}")
    json.dump(out, open(os.path.join(ROOT, "clinic", "member_ablation.json"), "w"), indent=1)
    print("\nRanking by cost of removal (most negative = most valuable):")
    for m, b in sorted(out.items(), key=lambda kv: kv[1]["mean"]["delta"]):
        sig = "significant" if (b["mean"]["ci95"][0] > 0) == (b["mean"]["ci95"][1] > 0) else "spans zero"
        print(f"   {m:<8} {b['mean']['delta']:+.4f}  ({sig})")
    print("\nwrote clinic/member_ablation.json")


if __name__ == "__main__":
    main()
