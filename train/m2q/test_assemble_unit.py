"""Unit test for the m2q test-path assembly (CPU, no GPU, no model download).

Fabricates 5 tiny fold npz + rows, then drives assemble() through:
 1. happy path: fold npz all carry test_probs -> m2q_<task>_test.npz written, mean correct;
 2. missing path: one fold lacks test_probs -> loud AssertionError naming the fold;
 3. no-test path: test_rows=None -> byte-identical legacy behavior, no test npz.
Run on the cluster inside the csa-m2 env (srun, CPU) or anywhere torch+numpy exist.
"""
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_m2q import assemble  # noqa: E402
from common import LABELS  # noqa: E402

TASK = "st3"
K = len(LABELS[TASK])
N_TR, N_DEV, N_TE = 20, 6, 7


def mk_rows(n, prefix, with_fold=False):
    rows = []
    for i in range(n):
        r = {"instanceID": f"{prefix}{i}"}
        if with_fold:
            r["fold"] = i % 5
        rows.append(r)
    return rows


def mk_workdir(with_test, drop_fold=None):
    wd = tempfile.mkdtemp(prefix="m2q-unit-")
    train_rows = mk_rows(N_TR, "tr", with_fold=True)
    for fold in range(5):
        va = [r for r in train_rows if r["fold"] == fold]
        kw = dict(
            oof_ids=np.array([r["instanceID"] for r in va]),
            oof_probs=np.random.rand(len(va), K).astype(np.float32),
            dev_probs=np.random.rand(N_DEV, K).astype(np.float32),
            oof_f1=0.5, dev_f1=0.5, seed=42 + fold)
        if with_test and fold != drop_fold:
            kw["test_probs"] = np.full((N_TE, K), 0.1 * (fold + 1), dtype=np.float32)
        np.savez(os.path.join(wd, f"fold{fold}.npz"), **kw)
    return wd, train_rows


def main():
    train_rows = None
    dev_rows = mk_rows(N_DEV, "de")
    test_rows = mk_rows(N_TE, "te")

    # 1. happy path
    wd, train_rows = mk_workdir(with_test=True)
    pd_ = tempfile.mkdtemp(prefix="m2q-preds-")
    assemble(TASK, train_rows, dev_rows, wd, pd_, test_rows)
    z = np.load(os.path.join(pd_, f"m2q_{TASK}_test.npz"), allow_pickle=True)
    assert list(z["ids"]) == [r["instanceID"] for r in test_rows]
    expect = np.mean([0.1 * (f + 1) for f in range(5)])
    assert np.allclose(z["probs"], expect), (z["probs"][0][0], expect)
    assert z["probs"].dtype == np.float32
    print("[1] happy path: test npz written, mean over folds correct")
    shutil.rmtree(wd); shutil.rmtree(pd_)

    # 2. missing test_probs in fold 3 -> loud failure naming the fold
    wd, train_rows = mk_workdir(with_test=True, drop_fold=3)
    pd_ = tempfile.mkdtemp(prefix="m2q-preds-")
    try:
        assemble(TASK, train_rows, dev_rows, wd, pd_, test_rows)
        raise SystemExit("FAIL: assemble accepted a fold without test_probs")
    except AssertionError as e:
        assert "3" in str(e), f"error does not name fold 3: {e}"
        print("[2] missing path: loud AssertionError naming fold 3")
    assert not os.path.exists(os.path.join(pd_, f"m2q_{TASK}_test.npz"))
    shutil.rmtree(wd); shutil.rmtree(pd_)

    # 3. legacy path: no test rows -> oof+dev only
    wd, train_rows = mk_workdir(with_test=False)
    pd_ = tempfile.mkdtemp(prefix="m2q-preds-")
    assemble(TASK, train_rows, dev_rows, wd, pd_, None)
    assert os.path.exists(os.path.join(pd_, f"m2q_{TASK}_oof.npz"))
    assert os.path.exists(os.path.join(pd_, f"m2q_{TASK}_dev.npz"))
    assert not os.path.exists(os.path.join(pd_, f"m2q_{TASK}_test.npz"))
    print("[3] legacy path: oof+dev written, no test npz")
    print("ASSEMBLE-UNIT PASS")


if __name__ == "__main__":
    main()
