"""Paired channel-cluster bootstrap for ChildSafeAds mean-macro-F1 deltas.

Uncertainty estimate used for every ablation keep-or-drop decision: resample channelIDs
with replacement (each channel carries all its instances), compute BOTH systems'
scores per resample, take the delta. Empty-class convention: a class with zero
gold positives in a resample is dropped from that resample's macro for both
systems (mirrors the pinned 'present' semantics); per-class drop rates logged.

API: bootstrap_delta(gold_rows, preds_a, preds_b, n=1000, seed=0) -> dict
"""
import json
import random
from collections import defaultdict

import numpy as np

from local_scorer import ST1, ST2, ST3


def _sets(gold_rows, preds):
    g = {t: {} for t in ("st1", "st2", "st3")}
    p = {t: {} for t in ("st1", "st2", "st3")}
    for iid, r in gold_rows.items():
        lab = r["labels"]
        g["st1"][iid] = {lab["st1"]}
        g["st2"][iid] = set(lab["st2"])
        g["st3"][iid] = set(lab["st3"])
        pr = preds.get(iid, {})
        p["st1"][iid] = {pr["st1"]} if "st1" in pr else set()
        p["st2"][iid] = set(pr.get("st2", []))
        p["st3"][iid] = set(pr.get("st3", []))
    return g, p


def _macro_present(gold_sets, pred_sets, label_space, iids):
    scores = []
    dropped = []
    for c in label_space:
        tp = fp = fn = 0
        for i in iids:
            gin, pin = c in gold_sets[i], c in pred_sets[i]
            if gin and pin:
                tp += 1
            elif pin:
                fp += 1
            elif gin:
                fn += 1
        if tp == fp == fn == 0:
            dropped.append(c)
            continue
        scores.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return (sum(scores) / len(scores) if scores else 0.0), dropped


def mean_macro(gold_sets, pred_sets, iids):
    drops = []
    total = 0.0
    for task, space in (("st1", ST1), ("st2", ST2), ("st3", ST3)):
        m, d = _macro_present(gold_sets[task], pred_sets[task], space, iids)
        total += m
        drops += [f"{task}:{c}" for c in d]
    return total / 3, drops


def bootstrap_delta(gold_rows, preds_a, preds_b, n=1000, seed=0):
    """gold_rows: iid->row (with labels + channel_context); preds_*: iid->pred obj."""
    ga, pa = _sets(gold_rows, preds_a)
    gb, pb = _sets(gold_rows, preds_b)
    by_ch = defaultdict(list)
    for iid, r in gold_rows.items():
        by_ch[r["channel_context"]["channelID"]].append(iid)
    channels = sorted(by_ch)
    rng = random.Random(seed)
    deltas = []
    drop_counter = defaultdict(int)
    for _ in range(n):
        sample = [c for _ in channels for c in [channels[rng.randrange(len(channels))]]]
        iids = [i for c in sample for i in by_ch[c]]
        ma, da = mean_macro(ga, pa, iids)
        mb, _ = mean_macro(gb, pb, iids)
        deltas.append(ma - mb)
        for d in da:
            drop_counter[d] += 1
    deltas = np.asarray(deltas)
    return {
        "delta_mean": float(deltas.mean()),
        "delta_se": float(deltas.std(ddof=1)),
        "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "class_drop_rates": {k: v / n for k, v in sorted(drop_counter.items()) if v / n > 0.01},
        "n": n,
    }


if __name__ == "__main__":
    import sys
    gold = {json.loads(l)["instanceID"]: json.loads(l) for l in open(sys.argv[1]) if l.strip()}
    a = {json.loads(l)["instanceID"]: json.loads(l) for l in open(sys.argv[2]) if l.strip()}
    b = {json.loads(l)["instanceID"]: json.loads(l) for l in open(sys.argv[3]) if l.strip()}
    print(json.dumps(bootstrap_delta(gold, a, b), indent=1))
