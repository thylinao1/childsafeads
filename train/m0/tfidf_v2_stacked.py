"""m0s: field-aware TF-IDF + fold-honest STACKED features (Mac CPU).

Upgrades over m0:
  - separate TF-IDF blocks per field group (transcript | title+description | page)
    so disclosure cues in the description are not drowned by page text;
  - official_disclosure as explicit one-hot features;
  - ST1/ST2 stacking into ST3 (and ST2 into ST1) using fold-honest OOF probs:
    fold k's ST3 model sees ST1/ST2 probabilities produced by models that never
    trained on fold k (the same OOF discipline as everything else).

Writes preds/m0s_{task}_{oof,dev}.npz per the standard contract.
"""
import json
import os
import sys

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "eval"))
from local_scorer import ST1, ST2, ST3  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SEED = 42


def fields_of(r):
    vc, cc, pp = r["video_context"], r["channel_context"], r.get("product_page") or {}
    disc = (vc.get("official_disclosure", "") or "unknown")
    return {
        "tr": r["transcript"]["text"] or "",
        "td": f"{vc.get('title','')}\n{vc.get('description','')}\nCHANNEL {cc.get('channel_name','')}",
        "pg": f"{pp.get('page_title','')}\n{pp.get('text','')}",
        "disc": disc,
    }


def disc_onehot(d):
    return [1.0 if d == "true" else 0.0, 1.0 if d == "false" else 0.0,
            1.0 if d not in ("true", "false") else 0.0]


def fit_lr_multilabel(space, X, ysets, Xva, Xde, Xte=None):
    """returns (val_probs, dev_probs, test_probs or None)"""
    pv = np.zeros((Xva.shape[0], len(space)), dtype=np.float32)
    pd = np.zeros((Xde.shape[0], len(space)), dtype=np.float32)
    pt = np.zeros((Xte.shape[0], len(space)), dtype=np.float32) if Xte is not None else None
    for j, c in enumerate(space):
        yj = np.array([1 if c in s else 0 for s in ysets])
        if yj.sum() == 0:
            continue
        clf = LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced", random_state=SEED)
        clf.fit(X, yj)
        pv[:, j] = clf.predict_proba(Xva)[:, 1]
        pd[:, j] = clf.predict_proba(Xde)[:, 1]
        if Xte is not None:
            pt[:, j] = clf.predict_proba(Xte)[:, 1]
    return pv, pd, pt


def fit_lr_single(space, X, ylabels, Xva, Xde, Xte=None):
    y = np.array([space.index(l) for l in ylabels])
    clf = LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced", random_state=SEED)
    clf.fit(X, y)
    pv = np.zeros((Xva.shape[0], len(space)), dtype=np.float32)
    pd = np.zeros((Xde.shape[0], len(space)), dtype=np.float32)
    pt = np.zeros((Xte.shape[0], len(space)), dtype=np.float32) if Xte is not None else None
    for j_local, j_global in enumerate(clf.classes_):
        pv[:, j_global] = clf.predict_proba(Xva)[:, j_local]
        pd[:, j_global] = clf.predict_proba(Xde)[:, j_local]
        if Xte is not None:
            pt[:, j_global] = clf.predict_proba(Xte)[:, j_local]
    return pv, pd, pt


def main():
    train = [json.loads(l) for l in open(f"{ROOT}/data/train.jsonl")]
    dev = [json.loads(l) for l in open(f"{ROOT}/data/dev.jsonl")]
    test = ([json.loads(l) for l in open(f"{ROOT}/data/test.jsonl")]
            if os.path.exists(f"{ROOT}/data/test.jsonl") else None)
    folds = json.load(open(f"{ROOT}/eval/folds.json"))
    fold_of = np.array([folds[r["channel_context"]["channelID"]] for r in train])

    Ftr = [fields_of(r) for r in train]
    Fde = [fields_of(r) for r in dev]
    Fte = [fields_of(r) for r in test] if test is not None else None
    y1 = [r["labels"]["st1"] for r in train]
    y2 = [set(r["labels"]["st2"]) for r in train]
    y3 = [set(r["labels"]["st3"]) for r in train]

    ids_tr = np.array([r["instanceID"] for r in train])
    ids_dev = np.array([r["instanceID"] for r in dev])
    ids_te = np.array([r["instanceID"] for r in test]) if test is not None else None

    n_tr, n_de = len(train), len(dev)
    oof = {t: np.zeros((n_tr, len(s)), dtype=np.float32) for t, s in
           (("st1", ST1), ("st2", ST2), ("st3", ST3))}
    devf = {t: np.zeros((5, n_de, len(s)), dtype=np.float32) for t, s in
            (("st1", ST1), ("st2", ST2), ("st3", ST3))}
    testf = ({t: np.zeros((5, len(test), len(s)), dtype=np.float32) for t, s in
              (("st1", ST1), ("st2", ST2), ("st3", ST3))} if test is not None else None)

    for k in range(5):
        tr_idx = np.where(fold_of != k)[0]
        va_idx = np.where(fold_of == k)[0]
        blocks_tr, blocks_va, blocks_de, blocks_te = [], [], [], []
        for fld, feats in (("tr", 40000), ("td", 40000), ("pg", 40000)):
            vw = TfidfVectorizer(ngram_range=(1, 2), max_features=feats, min_df=2, sublinear_tf=True)
            blocks_tr.append(vw.fit_transform([Ftr[i][fld] for i in tr_idx]))
            blocks_va.append(vw.transform([Ftr[i][fld] for i in va_idx]))
            blocks_de.append(vw.transform([f[fld] for f in Fde]))
            if test is not None:
                blocks_te.append(vw.transform([f[fld] for f in Fte]))
        vch = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=50000, min_df=2, sublinear_tf=True)
        blocks_tr.append(vch.fit_transform([Ftr[i]["tr"] + " " + Ftr[i]["td"] for i in tr_idx]))
        blocks_va.append(vch.transform([Ftr[i]["tr"] + " " + Ftr[i]["td"] for i in va_idx]))
        blocks_de.append(vch.transform([f["tr"] + " " + f["td"] for f in Fde]))
        if test is not None:
            blocks_te.append(vch.transform([f["tr"] + " " + f["td"] for f in Fte]))
        d_tr = csr_matrix(np.array([disc_onehot(Ftr[i]["disc"]) for i in tr_idx]))
        d_va = csr_matrix(np.array([disc_onehot(Ftr[i]["disc"]) for i in va_idx]))
        d_de = csr_matrix(np.array([disc_onehot(f["disc"]) for f in Fde]))
        Xtr = hstack(blocks_tr + [d_tr]).tocsr()
        Xva = hstack(blocks_va + [d_va]).tocsr()
        Xde = hstack(blocks_de + [d_de]).tocsr()
        if test is not None:
            d_te = csr_matrix(np.array([disc_onehot(f["disc"]) for f in Fte]))
            Xte = hstack(blocks_te + [d_te]).tocsr()
        else:
            Xte = None

        # stage 1: base ST1 + ST2 on text
        p1v, p1d, p1t = fit_lr_single(ST1, Xtr, [y1[i] for i in tr_idx], Xva, Xde, Xte)
        p2v, p2d, p2t = fit_lr_multilabel(ST2, Xtr, [y2[i] for i in tr_idx], Xva, Xde, Xte)
        oof["st1"][va_idx] = p1v; devf["st1"][k] = p1d
        oof["st2"][va_idx] = p2v; devf["st2"][k] = p2d
        if test is not None:
            testf["st1"][k] = p1t
            testf["st2"][k] = p2t

        # stage 2: ST3 with stacked ST1+ST2 probs (fold-honest: inner-CV OOF on the
        # training portion supplies the stack features for fitting)
        inner_p1 = np.zeros((len(tr_idx), len(ST1)), dtype=np.float32)
        inner_p2 = np.zeros((len(tr_idx), len(ST2)), dtype=np.float32)
        inner_fold = fold_of[tr_idx]
        for kk in set(inner_fold):
            a = np.where(inner_fold != kk)[0]
            b = np.where(inner_fold == kk)[0]
            q1, _, _ = fit_lr_single(ST1, Xtr[a], [y1[tr_idx[i]] for i in a], Xtr[b], Xtr[:1])
            q2, _, _ = fit_lr_multilabel(ST2, Xtr[a], [y2[tr_idx[i]] for i in a], Xtr[b], Xtr[:1])
            inner_p1[b] = q1
            inner_p2[b] = q2
        Str = hstack([Xtr, csr_matrix(inner_p1), csr_matrix(inner_p2)]).tocsr()
        Sva = hstack([Xva, csr_matrix(p1v), csr_matrix(p2v)]).tocsr()
        Sde = hstack([Xde, csr_matrix(p1d), csr_matrix(p2d)]).tocsr()
        Ste = (hstack([Xte, csr_matrix(p1t), csr_matrix(p2t)]).tocsr()
               if test is not None else None)
        p3v, p3d, p3t = fit_lr_multilabel(ST3, Str, [y3[i] for i in tr_idx], Sva, Sde, Ste)
        oof["st3"][va_idx] = p3v; devf["st3"][k] = p3d
        if test is not None:
            testf["st3"][k] = p3t
        print(f"fold {k} done", flush=True)

    for t in ("st1", "st2", "st3"):
        np.savez(f"{ROOT}/preds/m0s_{t}_oof.npz", ids=ids_tr, probs=oof[t])
        np.savez(f"{ROOT}/preds/m0s_{t}_dev.npz", ids=ids_dev, probs=devf[t].mean(axis=0),
                 dev_folds=devf[t])
        if test is not None:
            np.savez(f"{ROOT}/preds/m0s_{t}_test.npz", ids=ids_te,
                     probs=testf[t].mean(axis=0))
    print("m0s done")


if __name__ == "__main__":
    main()
