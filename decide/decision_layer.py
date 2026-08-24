"""ChildSafeAds decision layer: member blend -> macro-F1-optimal decisions.

Fits ONLY on pooled train-OOF (never dev). Produces decided predictions for any
split from stored npz probability files (contract: preds/<member>_<task>_<split>.npz,
ids + probs, class order = eval/local_scorer.py).

ST1: argmax over log p - tau * log(prior), tau grid on OOF macro-F1; tau=0 kept
     unless adjustment wins; classes in NEVER_PREDICT are masked out.
ST2/ST3: per-class threshold grid on pooled OOF; stability = per-fold refit spread;
     low-support classes fall back to Lipton t = F1_opt/2 (arXiv 1402.1892);
     sanity clip [0.01, 0.95] only.
ST3 constraints (after thresholding): T1.1 xor T1.2 (keep higher prob); substantive
     flag drops no_flag/insufficient_context; none fired -> the better housekeeping
     class by its own thresholded decision, default no_flag.

Usage:
  python decide/decision_layer.py fit  --members m0            # fit on OOF, report OOF+dev
  python decide/decision_layer.py emit --members m0 --split dev --out preds_dev.jsonl
"""
import argparse
import json
import os
import re
import sys

import numpy as np

# widened 2026-08-12: the original missed "thanks for sponsoring this video", a very common
# disclosure form (259 train rows). Verified against gold: 0 of those 259 are gold
# undisclosed_advertising, so the rule still has 0 violations in 2353 rows.
SPONSOR_RX = re.compile(r"\bsponsor(s|ed|ing|ship|ships)?\b")
# T1.1 names the description explicitly as one of the three places a disclosure can live,
# so a sponsorship phrase there means "nowhere disclosed" cannot hold. Verified against gold:
# 777 train rows match, 1 is gold undisclosed_advertising (0.13%).
DESC_DISCLOSURE_RX = re.compile(
    r"(sponsored by|sponsor of (this|today)|for sponsoring|sponsoring (this|today)"
    r"|this (video|episode) is sponsored|brought to you by|#ad\b|#sponsored"
    r"|paid promotion|paid partnership|in partnership with|our sponsor)", re.I)

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "eval"))
from local_scorer import ST1, ST2, ST3, score as score_files  # noqa: E402

NEVER_PREDICT_ST1 = ["other"]  # 2 train / 0 dev instances; 'present' semantics make firing pure downside
# per-class extra members mixed into the st3 blend column at equal weight
# (m3q32B sees age_restricted better than the blend; measured OOF +0.011 on that class)
# direct_exhortation: 54 of 137 OOF false negatives sit within a few hundredths of tau with
# m3q32A >= 0.7, i.e. a flat 5-member mean dilutes members that are right. Mixing s3s and
# m3q32A back in moves informative weight from 0.40 to 0.55 of the column.
ST3_COL_EXTRA = {"direct_exhortation": ["s3s", "m3q32A"]}
# Full column replacement was tested 2026-08-12 and rejected:
# paired 95% CI [-0.00002, +0.00496] at 4000 reps, P(better) 0.971, just under the bar.
# The mechanism is sound (zero-shot AUC 0.967 vs blend 0.921 on this class) so it is worth
# revisiting inside a larger bundle, but it does not stand alone. Empty = disabled.
ST3_COL_ONLY = {"age_restricted_or_prohibited_product": ["m3q32A", "m3q32B", "m3q32C", "m1m"]}
SUBSTANTIVE_ST3 = ["undisclosed_advertising", "inadequate_disclosure", "direct_exhortation",
                   "misleading_claim", "age_restricted_or_prohibited_product", "hfss_food_marketing"]
PARAMS_PATH = os.path.join(ROOT, "decide", "params.json")


def load_member(member, task, split):
    z = np.load(os.path.join(ROOT, "preds", f"{member}_{task}_{split}.npz"), allow_pickle=True)
    return [str(x) for x in z["ids"]], z["probs"].astype(np.float64)


def load_blend(members, task, split, weights=None):
    ids0, acc, w = None, None, None
    weights = weights or {m: 1.0 for m in members}
    for m in members:
        ids, p = load_member(m, task, split)
        if ids0 is None:
            ids0, acc, w = ids, np.zeros_like(p), 0.0
        assert ids == ids0, f"id order mismatch member {m} {task} {split}"
        acc += weights[m] * p
        w += weights[m]
    return ids0, acc / w


def gold_sets(path, task):
    out = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            lab = r["labels"][task]
            out[r["instanceID"]] = {lab} if task == "st1" else set(lab)
    return out


def f1_at(y, p, t):
    pred = p >= t
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum())
    if tp == 0 and (fp or fn):
        return 0.0
    if tp == fp == fn == 0:
        return None
    return 2 * tp / (2 * tp + fp + fn)


def fit_thresholds(space, ids, probs, gold, fold_of=None, min_support=15):
    grid = np.round(np.arange(0.01, 0.991, 0.01), 3)
    y_all = {c: np.array([c in gold[i] for i in ids]) for c in space}
    params = {}
    for j, c in enumerate(space):
        y = y_all[c]
        p = probs[:, j]
        support = int(y.sum())
        f1s = [(t, f1_at(y, p, t) or 0.0) for t in grid]
        t_best, f_best = max(f1s, key=lambda x: x[1])
        if support < min_support:
            t_use = max(0.5 * f_best, 0.05) if f_best > 0 else 0.98  # Lipton plug-in; near-dead class -> fire almost never
            src = "lipton"
        else:
            # stability: refit per fold, use spread
            if fold_of is not None:
                ts = []
                for k in range(5):
                    m = fold_of != k
                    if y[m].sum() >= 3:
                        ts.append(max(((t, f1_at(y[m], p[m], t) or 0.0) for t in grid), key=lambda x: x[1])[0])
                spread = (max(ts) - min(ts)) if len(ts) >= 2 else 0.0
                if spread > 0.35:  # unstable -> shrink toward Lipton, precision-leaning
                    t_use = max(t_best, 0.5 * f_best)
                    src = "shrunk"
                else:
                    t_use = t_best
                    src = "grid"
            else:
                t_use = t_best
                src = "grid"
        params[c] = {"t": float(np.clip(t_use, 0.01, 0.95)), "f1_oof": round(f_best, 4),
                     "support": support, "src": src}
    return params


def decide_multilabel(space, probs, params):
    out = []
    for row in probs:
        chosen = [c for j, c in enumerate(space) if row[j] >= params[c]["t"]]
        out.append(chosen)
    return out


def decide_st1(probs, tau, prior):
    logp = np.log(np.clip(probs, 1e-9, 1))
    adj = logp - tau * np.log(prior)[None, :]
    for c in NEVER_PREDICT_ST1:
        adj[:, ST1.index(c)] = -1e18
    return [ST1[j] for j in adj.argmax(axis=1)]


def apply_st1_other_override(ids1, st1_dec, split, tau):
    """Zero-shot rescue for st1 'other' (2 train gold; encoders rank it dead last, so the
    argmax never fires it and present-semantics eat a full 0.0 class slot whenever gold
    has it; the 0.6021 test st1 return is arithmetically consistent with exactly that).
    m3q32A p(other) >= tau overrides the blend argmax; tau > 1 disables. Measured OOF
    2026-08-11: fires 18-29 rows at tau 0.70-0.75, catches 1 of 2 gold (p=0.772), macro
    0.6369 -> 0.6449-0.6521; test fires the SAME 7 rows anywhere in that window. The
    knife edge (single catchable gold) was initially accepted on an asymmetric-payoff
    argument, then disabled (st1_other_tau = 1.01) after the paired bootstrap; see the
    fit-time comment below."""
    if tau > 1.0:
        return st1_dec
    mids, mp = load_member("m3q32A", "st1", split)
    assert mids == ids1
    col = mp[:, ST1.index("other")]
    return [("other" if col[i] >= tau else d) for i, d in enumerate(st1_dec)]


IC_PREEMPT_EXEMPT = {"misleading_claim", "direct_exhortation",
                     "age_restricted_or_prohibited_product", "hfss_food_marketing"}


def apply_st3_constraints(labels, probs_row, space, nf_tau=1.01, nf_margin=0.0, ic_tau=1.01,
                          ic_preempt_tau=1.01):
    """no_flag-aware decision (motivation: fallback-only logic scored no_flag F1 0.325).

    nf_tau: if p(no_flag) >= nf_tau, weak flags (fired but with prob below
            their threshold + nf_margin... implemented as: p(flag) < p(no_flag)
            + nf_margin) are suppressed; no_flag wins if nothing survives.
    ic_tau: if p(insufficient_context) >= ic_tau and no strong flag, emit it.
    ic_preempt_tau: gold-IC rows (empty/noise transcripts) look exactly like
            'ad with no disclosure', so UA/ID fire and the ic fallback branch
            is unreachable on them. Preempt AFTER constraints: if p(IC) clears
            this bar and no exempt strong flag was decided, output IC alone.
    Defaults (>1) disable all three, reproducing the legacy fallback behavior.
    """
    s = set(labels)
    i11, i12 = space.index("undisclosed_advertising"), space.index("inadequate_disclosure")
    if "undisclosed_advertising" in s and "inadequate_disclosure" in s:
        s.discard("inadequate_disclosure" if probs_row[i11] >= probs_row[i12] else "undisclosed_advertising")
    p_nf = probs_row[space.index("no_flag")]
    p_ic = probs_row[space.index("insufficient_context")]
    if p_nf >= nf_tau:
        s = {c for c in s if c in SUBSTANTIVE_ST3 and probs_row[space.index(c)] >= p_nf + nf_margin}
    else:
        s &= set(SUBSTANTIVE_ST3)
    if p_ic >= ic_preempt_tau and not (s & IC_PREEMPT_EXEMPT):
        return ["insufficient_context"]
    if not s and p_ic >= ic_tau:
        return ["insufficient_context"]
    if not s:
        return ["no_flag"] if p_nf >= p_ic or p_ic < ic_tau else ["insufficient_context"]
    return sorted(s)


SPLIT_RAW = {"oof": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}


def disclosure_map(split):
    out = {}
    for line in open(os.path.join(ROOT, "data", SPLIT_RAW[split])):
        if line.strip():
            r = json.loads(line)
            out[r["instanceID"]] = r["video_context"].get("official_disclosure", "")
    return out


def sponsor_map(split):
    out = {}
    for line in open(os.path.join(ROOT, "data", SPLIT_RAW[split])):
        if line.strip():
            r = json.loads(line)
            tr = (r["transcript"]["text"] or "").lower()
            out[r["instanceID"]] = bool(SPONSOR_RX.search(tr))
    return out


def desc_disclosure_map(split):
    out = {}
    for line in open(os.path.join(ROOT, "data", SPLIT_RAW[split])):
        if line.strip():
            r = json.loads(line)
            desc = (r.get("video_context") or {}).get("description") or ""
            out[r["instanceID"]] = bool(DESC_DISCLOSURE_RX.search(desc))
    return out


def mix_col_extra(ids3, p3, n_members, split):
    """Mix ST3_COL_EXTRA members into their target st3 columns at equal weight."""
    for cls, extras in ST3_COL_EXTRA.items():
        j = ST3.index(cls)
        acc = p3[:, j] * n_members
        for m in extras:
            ids_e, p_e = load_member(m, "st3", split)
            assert ids_e == ids3, f"id order mismatch extra member {m} st3 {split}"
            acc = acc + p_e[:, j]
        p3[:, j] = acc / (n_members + len(extras))
    return p3


def set_col_only(ids3, p3, split):
    """Overwrite an st3 column with an equal-weight mean of named members only."""
    for cls, members in ST3_COL_ONLY.items():
        j = ST3.index(cls)
        acc = np.zeros(len(ids3))
        for m in members:
            ids_e, p_e = load_member(m, "st3", split)
            assert ids_e == ids3, f"id order mismatch col-only member {m} st3 {split}"
            acc = acc + p_e[:, j]
        p3[:, j] = acc / len(members)
    return p3


def apply_hard_rules_st3(ids3, p3, split):
    """Zero p(undisclosed_advertising) where disclosure demonstrably exists.

    Rule 1 (0 violations / 2857 gold rows): official_disclosure == 'true'.
    Rule 2 (0 violations / 2857 gold rows): 'sponsor(s|ed|ing|ship)' spoken in the
    transcript IS a disclosure, so T1.1 (nowhere disclosed) cannot hold.
    Rule 3 (1 violation / 777 matching train rows): a sponsorship phrase in the video
    DESCRIPTION is likewise a disclosure available to the viewer; T1.1 names the
    description explicitly as one of the three places it looks.
    Applied in BOTH fit and emit so thresholds are fit on the same probs.
    """
    disc, spon = disclosure_map(split), sponsor_map(split)
    ddisc = desc_disclosure_map(split)
    j_t11 = ST3.index("undisclosed_advertising")
    for i, iid in enumerate(ids3):
        if disc.get(iid, "") == "true" or spon.get(iid, False) or ddisc.get(iid, False):
            p3[i, j_t11] = 0.0
    return p3


def emit_preds(members, split, params, weights=None):
    ids1, p1 = load_blend(members, "st1", split, weights)
    ids2, p2 = load_blend(members, "st2", split, weights)
    ids3, p3 = load_blend(members, "st3", split, weights)
    assert ids1 == ids2 == ids3
    p3 = mix_col_extra(ids3, p3, len(members), split)
    p3 = set_col_only(ids3, p3, split)
    p3 = apply_hard_rules_st3(ids3, p3, split)
    st1_dec = decide_st1(p1, params["st1_tau"], np.array(params["st1_prior"]))
    st1_dec = apply_st1_other_override(ids1, st1_dec, split, params.get("st1_other_tau", 1.01))
    st2_dec = decide_multilabel(ST2, p2, params["st2"])
    st3_raw = decide_multilabel(ST3, p3, params["st3"])
    nf = params.get("st3_nf", {"nf_tau": 1.01, "nf_margin": 0.0, "ic_tau": 1.01})
    rows = []
    for i, iid in enumerate(ids1):
        st2_i = st2_dec[i] or [ST2[int(np.argmax(p2[i]))]]  # never empty: best guess beats guaranteed miss
        st3_i = apply_st3_constraints(st3_raw[i], p3[i], ST3, nf["nf_tau"], nf["nf_margin"], nf["ic_tau"],
                                      nf.get("ic_preempt_tau", 1.01))
        rows.append({"instanceID": iid, "st1": st1_dec[i], "st2": sorted(st2_i), "st3": st3_i})
    return rows


def st3_macro_of(decided, gold3, ids):
    scores = []
    for c in ST3:
        tp = fp = fn = 0
        for i, iid in enumerate(ids):
            gin, pin = c in gold3[iid], c in decided[i]
            if gin and pin: tp += 1
            elif pin: fp += 1
            elif gin: fn += 1
        if tp == fp == fn == 0: continue
        scores.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(scores) / len(scores)


def fit_st3_nf(ids3, p3, params3, gold3):
    """Grid the no_flag-aware decision params on OOF ST3 macro."""
    raw = decide_multilabel(ST3, p3, params3)
    best = (None, -1.0)
    for nf_tau in [1.01, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]:
        for nf_margin in [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2]:
            if nf_tau > 1 and nf_margin != 0.0:
                continue
            # preemption ON disables the fallback ic_tau (it was pure noise:
            # OOF 15 fired / 2 TP); preemption OFF keeps the legacy ic grid
            for ic_preempt_tau, ic_taus in [(1.01, [1.01, 0.1, 0.15, 0.2, 0.3, 0.4]),
                                            (0.25, [1.01]), (0.29, [1.01]), (0.33, [1.01])]:
                for ic_tau in ic_taus:
                    dec = [apply_st3_constraints(raw[i], p3[i], ST3, nf_tau, nf_margin, ic_tau,
                                                 ic_preempt_tau)
                           for i in range(len(ids3))]
                    m = st3_macro_of(dec, gold3, ids3)
                    if m > best[1]:
                        best = ({"nf_tau": nf_tau, "nf_margin": nf_margin, "ic_tau": ic_tau,
                                 "ic_preempt_tau": ic_preempt_tau}, m)
    return best


def write_jsonl(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def score_split(rows, gold_path):
    tmp = os.path.join(ROOT, "decide", ".tmp_score.jsonl")
    write_jsonl(rows, tmp)
    r = score_files(gold_path, tmp, "present", 0)
    return {k: round(r[k], 4) for k in ("st1", "st2", "st3", "mean")}, r["per_class"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fit", "emit"])
    ap.add_argument("--members", required=True, help="comma-separated member ids")
    ap.add_argument("--weights", default=None, help="json dict member->weight")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", default=None)
    ap.add_argument("--params", default=PARAMS_PATH)
    args = ap.parse_args()
    members = args.members.split(",")
    weights = json.loads(args.weights) if args.weights else None

    train_gold_path = os.path.join(ROOT, "data", "train.jsonl")
    dev_gold_path = os.path.join(ROOT, "data", "dev.jsonl")

    if args.mode == "fit":
        folds = json.load(open(os.path.join(ROOT, "eval", "folds.json")))
        train_rows = [json.loads(l) for l in open(train_gold_path)]
        fold_of = np.array([folds[r["channel_context"]["channelID"]] for r in train_rows])

        ids1, p1 = load_blend(members, "st1", "oof", weights)
        g1 = gold_sets(train_gold_path, "st1")
        prior = np.array([max(sum(1 for i in ids1 if c in g1[i]), 0.5) for c in ST1]) / len(ids1)
        y1 = np.array([ST1.index(next(iter(g1[i]))) for i in ids1])
        best = (0.0, 0.0)
        for tau in np.round(np.arange(0.0, 1.51, 0.1), 2):
            dec = decide_st1(p1, tau, prior)
            # quick macro-F1 (present) on labels
            labs = sorted(set(y1) | {ST1.index(d) for d in dec})
            f1s = []
            for c in labs:
                yy = y1 == c
                pp = np.array([ST1.index(d) == c for d in dec])
                tp = int((yy & pp).sum()); fp = int((~yy & pp).sum()); fn = int((yy & ~pp).sum())
                if tp or fp or fn:
                    f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
            m = sum(f1s) / len(f1s)
            if m > best[1]:
                best = (tau, m)
        st1_tau, st1_oof = best

        # st1 'other' override selection on the same quick present-macro (off vs 0.70)
        def _quick_macro(dec):
            labs = sorted(set(y1) | {ST1.index(d) for d in dec})
            f1s = []
            for c in labs:
                yy = y1 == c
                pp = np.array([ST1.index(d) == c for d in dec])
                tp = int((yy & pp).sum()); fp = int((~yy & pp).sum()); fn = int((yy & ~pp).sum())
                if tp or fp or fn:
                    f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
            return sum(f1s) / len(f1s)
        # st1 'other' override: DISABLED after a paired channel bootstrap (2026-08-11).
        # Whole-OOF it looked like +0.0027 mean, but that estimate always contains both
        # gold 'other' rows. Paired bootstrap: delta -0.0043 mean, P(better) only 0.63,
        # which is just P(the one catchable gold channel is in the resample). Test-set
        # payoff is asymmetric the WRONG way: if test gold has no 'other', our 7 false
        # positives create a present-at-zero class slot worth -0.040 mean; if it does
        # have one and we catch it, +0.015. Break-even needs ~90% confidence that test
        # gold contains 'other', and the train base rate (2/2353 -> 0.43 expected in 503)
        # does not support that. Keep the function for the record; do not re-enable
        # without a real prevalence estimate.
        st1_other_tau = 1.01
        _ = _quick_macro  # kept for future st1 selection work

        ids2, p2 = load_blend(members, "st2", "oof", weights)
        params2 = fit_thresholds(ST2, ids2, p2, gold_sets(train_gold_path, "st2"), fold_of)
        ids3, p3 = load_blend(members, "st3", "oof", weights)
        p3 = mix_col_extra(ids3, p3, len(members), "oof")
        p3 = set_col_only(ids3, p3, "oof")
        p3 = apply_hard_rules_st3(ids3, p3, "oof")
        params3 = fit_thresholds(ST3, ids3, p3, gold_sets(train_gold_path, "st3"), fold_of)

        gold3 = gold_sets(train_gold_path, "st3")
        nf_params, nf_oof = fit_st3_nf(ids3, p3, params3, gold3)
        print(f"[st3-nf] tuned {nf_params} -> OOF st3 macro {nf_oof:.4f}")
        params = {"members": members, "weights": weights, "st1_tau": st1_tau,
                  "st1_other_tau": st1_other_tau,
                  "st1_prior": prior.tolist(), "st2": params2, "st3": params3,
                  "st3_nf": nf_params}
        json.dump(params, open(args.params, "w"), indent=1)

        oof_rows = emit_preds(members, "oof", params, weights)
        oof_scores, _ = score_split(oof_rows, train_gold_path)
        dev_rows = emit_preds(members, "dev", params, weights)
        dev_scores, dev_pc = score_split(dev_rows, dev_gold_path)
        print(json.dumps({"st1_tau": st1_tau, "st1_oof_macro": round(st1_oof, 4),
                          "OOF": oof_scores, "DEV": dev_scores}, indent=1))
        print("dev per-class ST3:", {c: round(v, 3) for c, v in dev_pc["st3"].items()})
        print("dev per-class ST2:", {c: round(v, 3) for c, v in dev_pc["st2"].items()})
        print("dev per-class ST1:", {c: round(v, 3) for c, v in dev_pc["st1"].items()})
    else:
        params = json.load(open(args.params))
        rows = emit_preds(members, args.split, params, weights)
        out = args.out or os.path.join(ROOT, f"preds_{args.split}.jsonl")
        write_jsonl(rows, out)
        print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
