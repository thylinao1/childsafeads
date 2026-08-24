"""s1s: fold-honest level-2 stacker for ST1 (sibling of stack_st3.py).

Motivation (2026-08-11): st1 blend argmax is weakly calibrated on the small classes
(none OOF F1 0.552 with 52 predicted vs 35 gold; physical_services 0.739). Features:
every member's st1 probs + mean st2/st3 probs (AUX) + the s3s aux flags/lengths.
Level-2 = multinomial logistic regression, same 5-fold channel grouping (row i is
predicted by a model that never saw fold(i)). class_weight balanced lifts the rare
classes; 'other' (2 gold) stays blocked at decision time by NEVER_PREDICT_ST1 and is
handled by the separate m3q32A override, so its huge balanced weight is harmless here.

Writes preds/s1s_st1_{oof,dev,test}.npz + EXACT-blend-mean st2/st3 passthroughs so
adding s1s to the blend changes only st1 (isolation pattern proven with s2s).
"""
import json
import os
import re
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

SPONSOR_RX = re.compile(r"\bsponsor(s|ed|ship)?\b")
ANY_DISC_RX = re.compile(
    r"\bsponsor(s|ed|ship)?\b|#ad\b|#advert|#sponsored|\bpartner(ed|ship)\b|\baffiliate"
    r"|\bsent me\b|\bfor sending\b|\buse (my|the|our) (code|link)\b"
    r"|\b(promo|discount|coupon) code\b|\bpaid (promotion|partnership)\b|\bgifted\b")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "eval"))
from local_scorer import ST1  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MEMBERS = ["m1d", "m1m", "m3q32A", "m3q32B", "m0s", "m0"]
AUX = ["m1d", "m1m", "m3q32A"]
BLEND = ["m1d", "m1m", "m3q32A", "m0s", "s3s"]  # for the exact passthroughs
SEED = 42


def load(member, task, split):
    z = np.load(f"{ROOT}/preds/{member}_{task}_{split}.npz", allow_pickle=True)
    return [str(x) for x in z["ids"]], z["probs"].astype(np.float64)


def aux_feats(rows):
    out = []
    for r in rows:
        d = r["video_context"].get("official_disclosure", "")
        pp = r.get("product_page") or {}
        tr = (r["transcript"]["text"] or "").lower()
        de = (r["video_context"].get("description") or "").lower()
        out.append([
            1.0 if d == "true" else 0.0, 1.0 if d == "false" else 0.0,
            1.0 if d not in ("true", "false") else 0.0,
            np.log1p(len(tr.split())),
            np.log1p(len(de.split())),
            np.log1p(len((pp.get("text") or "").split())),
            float(bool(SPONSOR_RX.search(tr))), float(bool(SPONSOR_RX.search(de))),
            float(bool(ANY_DISC_RX.search(tr))), float(bool(ANY_DISC_RX.search(de))),
        ])
    return np.array(out)


def build_X(split, rows):
    ids0 = None
    blocks = []
    for m in MEMBERS:
        ids, p = load(m, "st1", split)
        if ids0 is None:
            ids0 = ids
        assert ids == ids0, (m, split)
        blocks.append(p)
    for t, srcs in (("st2", AUX), ("st3", AUX)):
        acc = None
        for m in srcs:
            ids, p = load(m, t, split)
            assert ids == ids0
            acc = p if acc is None else acc + p
        blocks.append(acc / len(srcs))
    order = {r["instanceID"]: r for r in rows}
    blocks.append(aux_feats([order[i] for i in ids0]))
    return ids0, np.hstack(blocks)


def main():
    train_rows = [json.loads(l) for l in open(f"{ROOT}/data/train.jsonl")]
    dev_rows = [json.loads(l) for l in open(f"{ROOT}/data/dev.jsonl")]
    folds = json.load(open(f"{ROOT}/eval/folds.json"))

    ids_tr, Xtr = build_X("oof", train_rows)
    ids_dev, Xde = build_X("dev", dev_rows)
    test_rows = []
    if os.path.exists(f"{ROOT}/data/test.jsonl"):
        test_rows = [json.loads(l) for l in open(f"{ROOT}/data/test.jsonl")]
        ids_te, Xte = build_X("test", test_rows)
    gold = {r["instanceID"]: r["labels"]["st1"] for r in train_rows}
    fold_of = np.array([folds[r["channel_context"]["channelID"]]
                        for r in [ {rr["instanceID"]: rr for rr in train_rows}[i] for i in ids_tr ]])

    y = np.array([ST1.index(gold[i]) for i in ids_tr])
    n_cls = len(ST1)
    oof = np.zeros((len(ids_tr), n_cls), dtype=np.float32)
    for k in range(5):
        tr = fold_of != k
        va = fold_of == k
        clf = LogisticRegression(C=1.0, max_iter=4000, class_weight="balanced",
                                 random_state=SEED)
        clf.fit(Xtr[tr], y[tr])
        proba = clf.predict_proba(Xtr[va])
        for ci, c in enumerate(clf.classes_):
            oof[np.where(va)[0], c] = proba[:, ci]
    clf = LogisticRegression(C=1.0, max_iter=4000, class_weight="balanced",
                             random_state=SEED)
    clf.fit(Xtr, y)
    def full_proba(X, n):
        proba = clf.predict_proba(X)
        out = np.zeros((n, n_cls), dtype=np.float32)
        for ci, c in enumerate(clf.classes_):
            out[:, c] = proba[:, ci]
        return out
    devp = full_proba(Xde, len(ids_dev))
    assert np.isfinite(oof).all() and np.isfinite(devp).all()
    np.savez(f"{ROOT}/preds/s1s_st1_oof.npz", ids=np.array(ids_tr), probs=oof)
    np.savez(f"{ROOT}/preds/s1s_st1_dev.npz", ids=np.array(ids_dev), probs=devp)
    splits = [("oof", ids_tr), ("dev", ids_dev)]
    if test_rows:
        testp = full_proba(Xte, len(ids_te))
        assert np.isfinite(testp).all()
        np.savez(f"{ROOT}/preds/s1s_st1_test.npz", ids=np.array(ids_te), probs=testp)
        splits.append(("test", ids_te))
    # EXACT blend-mean passthroughs: adding s1s changes only the st1 column
    sys.path.insert(0, os.path.join(ROOT, "decide"))
    import decision_layer as dl
    for t in ("st2", "st3"):
        for sp, ids in splits:
            bids, bp = dl.load_blend(BLEND, t, sp)
            assert bids == list(ids)
            np.savez(f"{ROOT}/preds/s1s_{t}_{sp}.npz", ids=np.array(ids), probs=bp.astype(np.float32))
    print("s1s written:", ", ".join(s for s, _ in splits))


if __name__ == "__main__":
    main()
