"""s3g: gradient-boosted sibling of s3s, the ST3 level-2 stacker.

Why: s3s is a per-flag logistic regression over member probabilities plus a few aux
features. That is a tabular problem with roughly 66 columns and 2353 rows, which is
LightGBM territory: it can use interactions between members that a linear model cannot,
for example "flag it when the zero-shot model is confident AND the transcript is long AND
the platform disclosure label is false". ST3 carries most of the remaining error, so the
stacker is worth improving directly.

Identical contract and fold discipline to stack_st3.py: same 5-fold channel grouping, same
feature builder (imported, not duplicated, so the two stay in step), fold-honest OOF where
row i is predicted by a model that never saw fold(i). Writes preds/s3g_st3_{oof,dev,test}
plus st1/st2 passthroughs that are the exact canonical blend mean, so s3g can enter a blend
without touching the other two tasks.
"""
import json
import os
import sys

import numpy as np
from lightgbm import LGBMClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.join(ROOT, "decide"))
from local_scorer import ST3  # noqa: E402
import stack_st3 as base  # noqa: E402  (feature builder + MEMBERS live there)

SEED = 42
BLEND = ["m1d", "m1m", "m3q32A", "m0s", "s3s", "m2qS2"]  # for exact st1/st2 passthroughs
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, max_depth=4,
              min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
              reg_lambda=1.0, random_state=SEED, n_jobs=4, verbose=-1)


def fit_predict(Xtr, ytr, Xs):
    """One binary flag. class_weight balanced mirrors the LR stacker's setting."""
    clf = LGBMClassifier(class_weight="balanced", **PARAMS)
    clf.fit(Xtr, ytr)
    return [clf.predict_proba(X)[:, 1] for X in Xs]


def main():
    train_rows = [json.loads(l) for l in open(f"{ROOT}/data/train.jsonl")]
    dev_rows = [json.loads(l) for l in open(f"{ROOT}/data/dev.jsonl")]
    folds = json.load(open(f"{ROOT}/eval/folds.json"))

    ids_tr, Xtr = base.build_X("oof", train_rows)
    ids_dev, Xde = base.build_X("dev", dev_rows)
    test_rows, ids_te, Xte = [], None, None
    if os.path.exists(f"{ROOT}/data/test.jsonl"):
        test_rows = [json.loads(l) for l in open(f"{ROOT}/data/test.jsonl")]
        ids_te, Xte = base.build_X("test", test_rows)

    gold = {r["instanceID"]: set(r["labels"]["st3"]) for r in train_rows}
    by_id = {r["instanceID"]: r for r in train_rows}
    fold_of = np.array([folds[by_id[i]["channel_context"]["channelID"]] for i in ids_tr])
    Y = np.array([[1 if c in gold[i] else 0 for c in ST3] for i in ids_tr])
    print(f"s3g: X={Xtr.shape} folds={sorted(set(fold_of.tolist()))}")

    oof = np.zeros((len(ids_tr), len(ST3)), dtype=np.float32)
    for k in range(5):
        tr, va = fold_of != k, fold_of == k
        for j, c in enumerate(ST3):
            if Y[tr, j].sum() < 3:
                oof[va, j] = float(Y[tr, j].mean())
                continue
            oof[va, j] = fit_predict(Xtr[tr], Y[tr, j], [Xtr[va]])[0]
        print(f"  fold {k} done ({int(va.sum())} rows)")

    devp = np.zeros((len(ids_dev), len(ST3)), dtype=np.float32)
    testp = np.zeros((len(test_rows), len(ST3)), dtype=np.float32)
    for j, c in enumerate(ST3):
        if Y[:, j].sum() < 3:
            devp[:, j] = float(Y[:, j].mean())
            if test_rows:
                testp[:, j] = float(Y[:, j].mean())
            continue
        outs = fit_predict(Xtr, Y[:, j], [Xde] + ([Xte] if test_rows else []))
        devp[:, j] = outs[0]
        if test_rows:
            testp[:, j] = outs[1]

    assert np.isfinite(oof).all() and np.isfinite(devp).all()
    np.savez(f"{ROOT}/preds/s3g_st3_oof.npz", ids=np.array(ids_tr), probs=oof)
    np.savez(f"{ROOT}/preds/s3g_st3_dev.npz", ids=np.array(ids_dev), probs=devp)
    splits = [("oof", ids_tr), ("dev", ids_dev)]
    if test_rows:
        assert np.isfinite(testp).all()
        np.savez(f"{ROOT}/preds/s3g_st3_test.npz", ids=np.array(ids_te), probs=testp)
        splits.append(("test", ids_te))

    import decision_layer as dl
    for t in ("st1", "st2"):
        for sp, ids in splits:
            bids, bp = dl.load_blend(BLEND, t, sp)
            assert bids == list(ids)
            np.savez(f"{ROOT}/preds/s3g_{t}_{sp}.npz", ids=np.array(ids), probs=bp.astype(np.float32))
    print("s3g written:", ", ".join(s for s, _ in splits))


if __name__ == "__main__":
    main()
