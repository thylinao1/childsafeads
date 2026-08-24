"""Fit thresholds for the sample size we are actually scored on.

The decision layer picks each per-class threshold by maximising F1 on the pooled 2353-row
cross-validation set. We are scored on 503 rows. F1 is a non-linear function of the counts,
so the threshold that maximises F1 on a large pool is not in general the one that maximises
EXPECTED F1 on a small draw: with few positives, a handful of false positives costs more
than the same rate costs in the pool, which pushes the optimum toward precision.

This computes, for every class, the threshold maximising mean F1 over many channel-disjoint
503-row subsamples of the cross-validation set, and reports it against the pooled choice.
Channel-disjoint because the real test split is (verified: zero channel overlap with train).

  python clinic/expected_f1_thresholds.py            # report only
  python clinic/expected_f1_thresholds.py --write    # write decide/params_expf1.json
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.join(ROOT, "decide"))
from local_scorer import ST2, ST3  # noqa: E402
import decision_layer as dl  # noqa: E402

N_DRAW = 400
TEST_ROWS = 503
SEED = 20260812
GRID = np.round(np.arange(0.05, 0.951, 0.01), 3)


def draws(ids, chan, rng, n):
    by = {}
    for k, i in enumerate(ids):
        by.setdefault(chan[i], []).append(k)
    keys = list(by)
    out = []
    for _ in range(n):
        order = rng.permutation(len(keys))
        idx, tot = [], 0
        for k in order:
            idx += by[keys[k]]
            tot += len(by[keys[k]])
            if tot >= TEST_ROWS:
                break
        out.append(np.array(idx[:TEST_ROWS]))
    return out


def f1_counts(y, pred):
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum())
    if tp == 0:
        return 0.0 if (fp or fn) else None
    return 2 * tp / (2 * tp + fp + fn)


def main():
    write = "--write" in sys.argv
    params = json.load(open(os.path.join(ROOT, "decide", "params.json")))
    chan = {}
    for line in open(os.path.join(ROOT, "data", "train.jsonl")):
        r = json.loads(line)
        chan[r["instanceID"]] = r["channel_context"]["channelID"]
    rng = np.random.default_rng(SEED)
    out = {}
    for task, space in (("st2", ST2), ("st3", ST3)):
        ids, p = dl.load_blend(params["members"], task, "oof", params.get("weights"))
        if task == "st3":
            p = dl.mix_col_extra(ids, p, len(params["members"]), "oof")
            p = dl.set_col_only(ids, p, "oof")
            p = dl.apply_hard_rules_st3(ids, p, "oof")
        gold = dl.gold_sets(os.path.join(ROOT, "data", "train.jsonl"), task)
        sub = draws(ids, chan, rng, N_DRAW)
        print(f"\n=== {task} ===")
        print(f"{'class':38s} {'pooled t':>9s} {'expF1 t':>8s} {'pooled F1':>10s} "
              f"{'E[F1]@pooled':>13s} {'E[F1]@new':>10s}")
        out[task] = {}
        for j, c in enumerate(space):
            y = np.array([c in gold[i] for i in ids])
            col = p[:, j]
            pooled_t = params[task][c]["t"]
            # expected F1 over draws for every grid point
            best_t, best_e = pooled_t, -1.0
            e_at_pooled = None
            for t in GRID:
                pred = col >= t
                vals = [f1_counts(y[d], pred[d]) for d in sub]
                vals = [v for v in vals if v is not None]
                e = float(np.mean(vals)) if vals else 0.0
                if abs(t - pooled_t) < 1e-9:
                    e_at_pooled = e
                if e > best_e:
                    best_t, best_e = float(t), e
            if e_at_pooled is None:
                pred = col >= pooled_t
                vals = [f1_counts(y[d], pred[d]) for d in sub]
                vals = [v for v in vals if v is not None]
                e_at_pooled = float(np.mean(vals)) if vals else 0.0
            pooled_f1 = params[task][c]["f1_oof"]
            mark = "  *" if abs(best_t - pooled_t) > 0.02 and best_e > e_at_pooled + 0.002 else ""
            print(f"{c:38s} {pooled_t:9.2f} {best_t:8.2f} {pooled_f1:10.4f} "
                  f"{e_at_pooled:13.4f} {best_e:10.4f}{mark}")
            out[task][c] = {"t": best_t, "e_f1": round(best_e, 4),
                            "e_f1_at_pooled": round(e_at_pooled, 4)}
    if write:
        json.dump(out, open(os.path.join(ROOT, "decide", "params_expf1.json"), "w"), indent=1)
        print("\nwrote decide/params_expf1.json")


if __name__ == "__main__":
    main()
