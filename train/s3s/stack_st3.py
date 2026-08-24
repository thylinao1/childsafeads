"""s3s: fold-honest level-2 stacker for ST3.

Features per instance: every member's st3 probs (OOF for train rows, dev preds
for dev rows) + mean st1/st2 probs + official_disclosure one-hot + log lengths.
Level-2 = per-flag logistic regression, trained with the SAME 5-fold channel
grouping: stack-OOF row i is predicted by an L2 model that never saw fold(i).
Dev preds come from an L2 model trained on all stack-OOF rows.

Writes preds/s3s_st3_{oof,dev}.npz (contract-standard).
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
from local_scorer import ST3  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MEMBERS = ["m1d", "m1m", "m3q32A", "m3q32B", "m3q32C", "m0s", "m0"]
AUX = ["m1d", "m1m", "m3q32A"]  # st1/st2 signal sources
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
        ids, p = load(m, "st3", split)
        if ids0 is None:
            ids0 = ids
        assert ids == ids0, (m, split)
        blocks.append(p)
    for t, srcs in (("st1", AUX), ("st2", AUX)):
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
    gold = {r["instanceID"]: set(r["labels"]["st3"]) for r in train_rows}
    fold_of = np.array([folds[r["channel_context"]["channelID"]]
                        for r in [ {rr["instanceID"]: rr for rr in train_rows}[i] for i in ids_tr ]])

    Y = np.array([[1 if c in gold[i] else 0 for c in ST3] for i in ids_tr])
    oof = np.zeros((len(ids_tr), len(ST3)), dtype=np.float32)
    for k in range(5):
        tr = fold_of != k
        va = fold_of == k
        for j, c in enumerate(ST3):
            if Y[tr, j].sum() < 3:
                oof[va, j] = float(Y[tr, j].mean())
                continue
            clf = LogisticRegression(C=1.0, max_iter=4000, class_weight="balanced",
                                     random_state=SEED)
            clf.fit(Xtr[tr], Y[tr, j])
            oof[va, j] = clf.predict_proba(Xtr[va])[:, 1]
    devp = np.zeros((len(ids_dev), len(ST3)), dtype=np.float32)
    testp = np.zeros((len(test_rows), len(ST3)), dtype=np.float32)
    for j, c in enumerate(ST3):
        if Y[:, j].sum() < 3:
            devp[:, j] = float(Y[:, j].mean())
            if test_rows:
                testp[:, j] = float(Y[:, j].mean())
            continue
        clf = LogisticRegression(C=1.0, max_iter=4000, class_weight="balanced",
                                 random_state=SEED)
        clf.fit(Xtr, Y[:, j])
        devp[:, j] = clf.predict_proba(Xde)[:, 1]
        if test_rows:
            testp[:, j] = clf.predict_proba(Xte)[:, 1]
    assert np.isfinite(oof).all() and np.isfinite(devp).all()
    np.savez(f"{ROOT}/preds/s3s_st3_oof.npz", ids=np.array(ids_tr), probs=oof)
    np.savez(f"{ROOT}/preds/s3s_st3_dev.npz", ids=np.array(ids_dev), probs=devp)
    splits = [("oof", ids_tr), ("dev", ids_dev)]
    if test_rows:
        assert np.isfinite(testp).all()
        np.savez(f"{ROOT}/preds/s3s_st3_test.npz", ids=np.array(ids_te), probs=testp)
        splits.append(("test", ids_te))
    # neutral st1/st2 passthrough so s3s can enter blends without touching them:
    for t in ("st1", "st2"):
        for sp, ids in splits:
            acc = None
            for m in AUX:
                _, p = load(m, t, sp)
                acc = p if acc is None else acc + p
            np.savez(f"{ROOT}/preds/s3s_{t}_{sp}.npz", ids=np.array(ids), probs=(acc/len(AUX)).astype(np.float32))
    print("s3s written:", ", ".join(s for s, _ in splits))


if __name__ == "__main__":
    main()
