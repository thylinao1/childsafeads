"""Data-access-level ablation for the zero-shot lane (m3q32A, Qwen3-32B, prompt variant A).

The shared task ships four cumulative data access levels. `build_user_prompt(row, level=L)`
in train/m3q32/prompts.py truncates the serialised instance to level L, so a level is a
prompt field set and nothing else: no retraining, inference only.

    L1  transcript only
    L2  + video context (title, description, YouTube paid-promotion label)
    L3  + channel context (channel name)
    L4  + product page (resolved url, page title, page text)   <- what every shipped run used

Design. The comparison is member-level and deliberately stripped: one member, no blend, no
column mixing, no hard rules. Only the prompt's field set differs between arms. Decision
parameters are refitted independently at every level on the same out-of-fold predictions, so
each level is scored with thresholds calibrated to its own probabilities rather than to
level 4's. Scoring uses the pinned present-label-set convention.

Uncertainty is a paired channel-cluster bootstrap: channels are resampled with replacement
and both arms are scored on the identical draw, which is the same test every keep-or-drop
decision in this project was made under.

Usage:  python3 clinic/level_ablation.py [--reps 2000]
Writes clinic/level_ablation.json and prints the tables.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.join(ROOT, "decide"))

from local_scorer import ST1, ST2, ST3, score as score_files          # noqa: E402
from cluster_bootstrap import _sets, mean_macro, _macro_present        # noqa: E402
import decision_layer as DL                                            # noqa: E402

LEVELS = [1, 2, 3, 4]
MEMBER_OF = {1: "m3q32AL1", 2: "m3q32AL2", 3: "m3q32AL3", 4: "m3q32A"}
LEVEL_NAME = {1: "L1 transcript",
              2: "L2 +video context",
              3: "L3 +channel name",
              4: "L4 +product page"}
TRAIN_GOLD = os.path.join(ROOT, "data", "train.jsonl")
DEV_GOLD = os.path.join(ROOT, "data", "dev.jsonl")


def fit_level(member):
    """Fit ST1 tau/prior and per-class ST2/ST3 thresholds for one member on OOF only."""
    folds = json.load(open(os.path.join(ROOT, "eval", "folds.json")))
    train_rows = [json.loads(l) for l in open(TRAIN_GOLD) if l.strip()]
    fold_of = np.array([folds[r["channel_context"]["channelID"]] for r in train_rows])

    ids1, p1 = DL.load_member(member, "st1", "oof")
    g1 = DL.gold_sets(TRAIN_GOLD, "st1")
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

    ids2, p2 = DL.load_member(member, "st2", "oof")
    params2 = DL.fit_thresholds(ST2, ids2, p2, DL.gold_sets(TRAIN_GOLD, "st2"), fold_of)
    ids3, p3 = DL.load_member(member, "st3", "oof")
    params3 = DL.fit_thresholds(ST3, ids3, p3, DL.gold_sets(TRAIN_GOLD, "st3"), fold_of)

    return {"member": member, "st1_tau": best[0], "st1_prior": prior.tolist(),
            "st2": params2, "st3": params3}


def emit_level(member, split, params):
    """Member-only emission. No blend, no column mixing, no hard rules."""
    ids1, p1 = DL.load_member(member, "st1", split)
    ids2, p2 = DL.load_member(member, "st2", split)
    ids3, p3 = DL.load_member(member, "st3", split)
    assert ids1 == ids2 == ids3
    st1_dec = DL.decide_st1(p1, params["st1_tau"], np.array(params["st1_prior"]))
    st2_dec = DL.decide_multilabel(ST2, p2, params["st2"])
    st3_dec = DL.decide_multilabel(ST3, p3, params["st3"])
    rows = []
    for i, iid in enumerate(ids1):
        st2_i = st2_dec[i] or [ST2[int(np.argmax(p2[i]))]]
        st3_i = st3_dec[i] or [ST3[int(np.argmax(p3[i]))]]
        rows.append({"instanceID": iid, "st1": st1_dec[i],
                     "st2": sorted(st2_i), "st3": sorted(st3_i)})
    return rows


def score_rows(rows, gold_path, tag):
    out = os.path.join(ROOT, "clinic", f".lvl_{tag}.jsonl")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    r = score_files(gold_path, out, "present", 0)
    return {k: r[k] for k in ("st1", "st2", "st3", "mean")}, r["per_class"], out


def _indicator_blocks(gold_rows, preds, ids_order):
    """Per-task (TP, FP, FN) boolean matrices, rows in ids_order, columns = label space."""
    out = {}
    for task, space in (("st1", ST1), ("st2", ST2), ("st3", ST3)):
        n, k = len(ids_order), len(space)
        G = np.zeros((n, k), dtype=bool)
        P = np.zeros((n, k), dtype=bool)
        for i, iid in enumerate(ids_order):
            lab = gold_rows[iid]["labels"][task]
            gset = {lab} if task == "st1" else set(lab)
            pr = preds.get(iid, {})
            pset = ({pr["st1"]} if "st1" in pr else set()) if task == "st1" else set(pr.get(task, []))
            for j, c in enumerate(space):
                G[i, j] = c in gset
                P[i, j] = c in pset
        out[task] = (G & P, (~G) & P, G & (~P))
    return out


def _channel_agg(blocks, ids_order, gold_rows):
    """Collapse row-level indicators to per-channel sums so a resample is one matmul."""
    ch_of = [gold_rows[i]["channel_context"]["channelID"] for i in ids_order]
    channels = sorted(set(ch_of))
    cidx = {c: j for j, c in enumerate(channels)}
    rows_ch = np.array([cidx[c] for c in ch_of])
    agg = {}
    for task, (TP, FP, FN) in blocks.items():
        nC = len(channels)
        a = []
        for M in (TP, FP, FN):
            acc = np.zeros((nC, M.shape[1]))
            np.add.at(acc, rows_ch, M.astype(np.float64))
            a.append(acc)
        agg[task] = a
    return channels, agg


def _macro_from_counts(tp, fp, fn):
    """Present-label-set macro-F1 from per-class counts, vectorised over replicates.

    tp/fp/fn: (reps, n_classes). A class with tp=fp=fn=0 is dropped from that
    replicate's average, mirroring the pinned convention.
    """
    present = (tp + fp + fn) > 0
    denom = 2 * tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)
    f1 = np.where(present, f1, 0.0)
    npres = present.sum(axis=1)
    return np.where(npres > 0, f1.sum(axis=1) / np.maximum(npres, 1), 0.0)


def paired_bootstrap(gold_rows, preds_a, preds_b, reps, seed=0):
    """Paired channel-cluster bootstrap, vectorised. Both arms scored on the identical draw."""
    ids_order = sorted(gold_rows)
    ba = _indicator_blocks(gold_rows, preds_a, ids_order)
    bb = _indicator_blocks(gold_rows, preds_b, ids_order)
    channels, agg_a = _channel_agg(ba, ids_order, gold_rows)
    _, agg_b = _channel_agg(bb, ids_order, gold_rows)
    nC = len(channels)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, nC, size=(reps, nC))
    counts = np.zeros((reps, nC))
    for r in range(reps):
        counts[r] = np.bincount(draws[r], minlength=nC)

    cols = {}
    tot_a = np.zeros(reps); tot_b = np.zeros(reps)
    for task in ("st1", "st2", "st3"):
        ma = _macro_from_counts(*[counts @ M for M in agg_a[task]])
        mb = _macro_from_counts(*[counts @ M for M in agg_b[task]])
        cols[task] = ma - mb
        tot_a += ma; tot_b += mb
    cols["mean"] = tot_a / 3 - tot_b / 3

    out = {}
    for k, d in cols.items():
        out[k] = {"delta": float(d.mean()), "se": float(d.std(ddof=1)),
                  "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                  "p_better": float((d > 0).mean())}
    out["n_channels"] = nC
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()

    train_gold = {json.loads(l)["instanceID"]: json.loads(l)
                  for l in open(TRAIN_GOLD) if l.strip()}

    fitted, scores, predfiles = {}, {}, {}
    for L in LEVELS:
        m = MEMBER_OF[L]
        p = fit_level(m)
        fitted[L] = p
        oof_rows = emit_level(m, "oof", p)
        dev_rows = emit_level(m, "dev", p)
        s_oof, pc_oof, f_oof = score_rows(oof_rows, TRAIN_GOLD, f"L{L}_oof")
        s_dev, pc_dev, _ = score_rows(dev_rows, DEV_GOLD, f"L{L}_dev")
        scores[L] = {"oof": s_oof, "dev": s_dev,
                     "oof_per_class": pc_oof, "dev_per_class": pc_dev}
        predfiles[L] = f_oof
        print(f"[fit] L{L} ({m}) st1_tau={p['st1_tau']}  "
              f"OOF mean {s_oof['mean']:.4f} (st1 {s_oof['st1']:.4f} / st2 {s_oof['st2']:.4f} / st3 {s_oof['st3']:.4f})  "
              f"DEV mean {s_dev['mean']:.4f}")

    loaded = {L: {json.loads(l)["instanceID"]: json.loads(l)
                  for l in open(predfiles[L]) if l.strip()} for L in LEVELS}

    print(f"\n[bootstrap] paired channel-cluster, {args.reps} replicates, 632 channels, OOF")
    pairs = [(2, 1), (3, 2), (4, 3), (4, 1), (4, 2)]
    boots = {}
    for hi, lo in pairs:
        b = paired_bootstrap(train_gold, loaded[hi], loaded[lo], args.reps)
        boots[f"L{hi}-L{lo}"] = b
        mm = b["mean"]
        print(f"  L{hi} - L{lo}: mean {mm['delta']:+.4f} "
              f"se {mm['se']:.4f} CI [{mm['ci95'][0]:+.4f}, {mm['ci95'][1]:+.4f}] "
              f"P(better) {mm['p_better']:.3f}")
        for t in ("st1", "st2", "st3"):
            c = b[t]
            print(f"      {t}: {c['delta']:+.4f} CI [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] P {c['p_better']:.3f}")

    json.dump({"scores": {str(k): v for k, v in scores.items()},
               "bootstrap": boots,
               "fitted_params": {str(k): v for k, v in fitted.items()},
               "reps": args.reps},
              open(os.path.join(ROOT, "clinic", "level_ablation.json"), "w"), indent=1)
    print("\nwrote clinic/level_ablation.json")


if __name__ == "__main__":
    main()
