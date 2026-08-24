"""m0: TF-IDF + logistic regression baseline with fold-honest OOF (Mac, CPU, minutes).

Purpose: (1) an immediate honest floor; (2) exercises the full preds->decide->score
path; (3) a cheap decorrelated ensemble member.

Writes preds/m0_{st1,st2,st3}_{oof,dev}.npz per the standard contract:
  ids: instanceID strings; probs: float32 N x K, class order = eval/local_scorer.py.
"""
import json
import os
import sys

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "eval"))
from local_scorer import ST1, ST2, ST3  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SEED = 42


def text_of(r):
    vc, cc, pp = r["video_context"], r["channel_context"], r.get("product_page") or {}
    disc = vc.get("official_disclosure", "") or "unknown"
    return (f"TRANSCRIPT: {r['transcript']['text']}\nTITLE: {vc.get('title','')}\n"
            f"DESCRIPTION: {vc.get('description','')}\nPAIDLABEL_{disc}\n"
            f"CHANNEL: {cc.get('channel_name','')}\nPAGETITLE: {pp.get('page_title','')}\n"
            f"PAGE: {pp.get('text','')}")


def main():
    train = [json.loads(l) for l in open(f"{ROOT}/data/train.jsonl")]
    dev = [json.loads(l) for l in open(f"{ROOT}/data/dev.jsonl")]
    test = ([json.loads(l) for l in open(f"{ROOT}/data/test.jsonl")]
            if os.path.exists(f"{ROOT}/data/test.jsonl") else None)
    folds = json.load(open(f"{ROOT}/eval/folds.json"))
    fold_of = np.array([folds[r["channel_context"]["channelID"]] for r in train])

    Xtr_txt = [text_of(r) for r in train]
    Xdev_txt = [text_of(r) for r in dev]
    Xte_txt = [text_of(r) for r in test] if test is not None else None

    os.makedirs(f"{ROOT}/preds", exist_ok=True)
    ids_tr = np.array([r["instanceID"] for r in train])
    ids_dev = np.array([r["instanceID"] for r in dev])
    ids_te = np.array([r["instanceID"] for r in test]) if test is not None else None

    tasks = {
        "st1": (ST1, [r["labels"]["st1"] for r in train], False),
        "st2": (ST2, [set(r["labels"]["st2"]) for r in train], True),
        "st3": (ST3, [set(r["labels"]["st3"]) for r in train], True),
    }

    for task, (space, ytr, multilabel) in tasks.items():
        K = len(space)
        oof = np.zeros((len(train), K), dtype=np.float32)
        devp_folds = np.zeros((5, len(dev), K), dtype=np.float32)
        testp_folds = (np.zeros((5, len(test), K), dtype=np.float32)
                       if test is not None else None)
        for k in range(5):
            tr_idx = np.where(fold_of != k)[0]
            va_idx = np.where(fold_of == k)[0]
            vw = TfidfVectorizer(ngram_range=(1, 2), max_features=60000, min_df=2, sublinear_tf=True)
            vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=60000, min_df=2, sublinear_tf=True)
            Xtr = hstack([vw.fit_transform([Xtr_txt[i] for i in tr_idx]),
                          vc.fit_transform([Xtr_txt[i] for i in tr_idx])]).tocsr()
            Xva = hstack([vw.transform([Xtr_txt[i] for i in va_idx]),
                          vc.transform([Xtr_txt[i] for i in va_idx])]).tocsr()
            Xde = hstack([vw.transform(Xdev_txt), vc.transform(Xdev_txt)]).tocsr()
            Xte = (hstack([vw.transform(Xte_txt), vc.transform(Xte_txt)]).tocsr()
                   if test is not None else None)
            if multilabel:
                for j, c in enumerate(space):
                    yj = np.array([1 if c in ytr[i] else 0 for i in tr_idx])
                    if yj.sum() == 0:
                        oof[va_idx, j] = 0.0
                        devp_folds[k, :, j] = 0.0
                        if test is not None:
                            testp_folds[k, :, j] = 0.0
                        continue
                    clf = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced",
                                             random_state=SEED)
                    clf.fit(Xtr, yj)
                    oof[va_idx, j] = clf.predict_proba(Xva)[:, 1]
                    devp_folds[k, :, j] = clf.predict_proba(Xde)[:, 1]
                    if test is not None:
                        testp_folds[k, :, j] = clf.predict_proba(Xte)[:, 1]
            else:
                y = np.array([space.index(ytr[i]) for i in tr_idx])
                clf = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced",
                                         random_state=SEED)
                clf.fit(Xtr, y)
                proba = clf.predict_proba(Xva)
                probd = clf.predict_proba(Xde)
                probt = clf.predict_proba(Xte) if test is not None else None
                for j_local, j_global in enumerate(clf.classes_):
                    oof[va_idx, j_global] = proba[:, j_local]
                    devp_folds[k, :, j_global] = probd[:, j_local]
                    if test is not None:
                        testp_folds[k, :, j_global] = probt[:, j_local]
            print(f"{task} fold {k} done", flush=True)
        devp = devp_folds.mean(axis=0)
        np.savez(f"{ROOT}/preds/m0_{task}_oof.npz", ids=ids_tr, probs=oof)
        np.savez(f"{ROOT}/preds/m0_{task}_dev.npz", ids=ids_dev, probs=devp,
                 dev_folds=devp_folds)
        if test is not None:
            np.savez(f"{ROOT}/preds/m0_{task}_test.npz", ids=ids_te,
                     probs=testp_folds.mean(axis=0))
    print("m0 done")


if __name__ == "__main__":
    main()
