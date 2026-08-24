"""How big is a real gain? Channel-bootstrap std.err of the OOF metric.

Written 2026-08-11 after a pure variance-reduction change (a 4th DeBERTa seed) moved
OOF st1 by -0.005: that is the signature of comparing inside the noise band. The
working rule said a gain must exceed 1 std.err, but the std.err was never
measured, so every third-decimal keep-or-drop decision rested on an unquantified bar.

Method: emit OOF decisions ONCE with the canonical params (no refitting per replicate,
so this measures sampling noise of the estimate, not tuning noise), then resample
CHANNELS with replacement (channels, not rows: rows within a channel are correlated,
and the eval test set is channel-disjoint, so the channel is the exchangeable unit).
Recompute present-semantics macro-F1 per replicate with the pinned scorer internals.

Also reports a paired bootstrap when given two prediction files: the std.err of the
DIFFERENCE, which is the number a keep-or-drop decision actually needs (paired resampling
cancels the shared channel draw, so it is much tighter than comparing two marginals).

Usage:
  python clinic/oof_noise.py                          # noise band of the canonical config
  python clinic/oof_noise.py A.jsonl B.jsonl          # paired: is B - A real?
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.join(ROOT, "decide"))
from local_scorer import ST1, ST2, ST3, macro_f1  # noqa: E402

N_BOOT = 2000
SEED = 20260811


def gold_maps(path):
    g1, g2, g3, chan = {}, {}, {}, {}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        i = r["instanceID"]
        g1[i] = {r["labels"]["st1"]}
        g2[i] = set(r["labels"]["st2"])
        g3[i] = set(r["labels"]["st3"])
        chan[i] = r["channel_context"]["channelID"]
    return g1, g2, g3, chan


def pred_maps(rows):
    p1 = {r["instanceID"]: {r["st1"]} for r in rows}
    p2 = {r["instanceID"]: set(r["st2"]) for r in rows}
    p3 = {r["instanceID"]: set(r["st3"]) for r in rows}
    return p1, p2, p3


def score_subset(ids, golds, preds):
    """present-semantics macro-F1 per task over a multiset of instance ids."""
    out = []
    for (g, p, space) in zip(golds, preds, (ST1, ST2, ST3)):
        gs = {}
        ps = {}
        for n, i in enumerate(ids):  # unique key per draw so duplicates count twice
            k = f"{i}#{n}"
            gs[k] = g[i]
            ps[k] = p.get(i, set())
        s, _ = macro_f1(gs, ps, space, "present", 0)
        out.append(s)
    return out + [sum(out) / 3]


def bootstrap(ids_by_chan, golds, preds_list, n_boot=N_BOOT):
    rng = np.random.default_rng(SEED)
    chans = list(ids_by_chan)
    reps = [[] for _ in preds_list]
    for _ in range(n_boot):
        draw = rng.choice(len(chans), size=len(chans), replace=True)
        ids = [i for k in draw for i in ids_by_chan[chans[k]]]
        for s, preds in enumerate(preds_list):
            reps[s].append(score_subset(ids, golds, preds))
    return [np.array(r) for r in reps]


def load_rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main():
    import decision_layer as dl
    train_gold = os.path.join(ROOT, "data", "train.jsonl")
    g1, g2, g3, chan = gold_maps(train_gold)
    golds = (g1, g2, g3)

    if len(sys.argv) == 3:
        rows_a, rows_b = load_rows(sys.argv[1]), load_rows(sys.argv[2])
        label_a, label_b = os.path.basename(sys.argv[1]), os.path.basename(sys.argv[2])
    else:
        params = json.load(open(os.path.join(ROOT, "decide", "params.json")))
        rows_a = dl.emit_preds(params["members"], "oof", params, params.get("weights"))
        rows_b = None
        label_a, label_b = "canonical", None

    ids_by_chan = {}
    for i, c in chan.items():
        ids_by_chan.setdefault(c, []).append(i)
    preds_list = [pred_maps(rows_a)] + ([pred_maps(rows_b)] if rows_b else [])
    reps = bootstrap(ids_by_chan, golds, preds_list)

    names = ["st1", "st2", "st3", "mean"]
    print(f"channels={len(ids_by_chan)} rows={len(chan)} bootstrap={N_BOOT}\n")
    print(f"{label_a}:")
    for j, n in enumerate(names):
        col = reps[0][:, j]
        print(f"  {n:5s} mean={col.mean():.4f}  std.err={col.std(ddof=1):.4f}  "
              f"95% CI=[{np.percentile(col, 2.5):.4f}, {np.percentile(col, 97.5):.4f}]")
    if rows_b:
        print(f"\n{label_b} minus {label_a} (paired):")
        for j, n in enumerate(names):
            d = reps[1][:, j] - reps[0][:, j]
            frac = float((d > 0).mean())
            print(f"  {n:5s} delta={d.mean():+.4f}  std.err={d.std(ddof=1):.4f}  "
                  f"95% CI=[{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]  "
                  f"P(better)={frac:.2f}")


if __name__ == "__main__":
    main()
