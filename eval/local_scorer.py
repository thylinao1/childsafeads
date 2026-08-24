"""ChildSafeAds local scorer (rebuild-exact) with convention-grid pinning.

Metric: mean_macro_f1 = mean(st1_macro_f1, st2_macro_f1, st3_macro_f1).
The official scoring program is private; conventions are pinned empirically
against the organizer's majority-baseline leaderboard row (4dp columns):
    ST1 0.1510 / ST2 0.0422 / ST3 0.0851 / mean 0.0928

Usage:
    python eval/local_scorer.py score  <gold.jsonl> <preds.jsonl> [--convention NAME]
    python eval/local_scorer.py pin    <gold.jsonl> <preds.jsonl>   # enumerate grid vs pin targets
Exit nonzero on any self-check failure. Deterministic: pure function of inputs.
"""
import json
import sys

ST1 = ["physical_goods", "digital_content_or_services", "physical_services", "none", "other"]
ST2 = ["toys", "food", "apps", "hardware_electronics", "fashion", "health", "education",
       "financial", "gambling", "gambling_adjacent", "creator_community", "other"]
ST3 = ["undisclosed_advertising", "inadequate_disclosure", "direct_exhortation",
       "misleading_claim", "age_restricted_or_prohibited_product", "hfss_food_marketing",
       "no_flag", "insufficient_context"]

PIN = {"st1": 0.1510, "st2": 0.0422, "st3": 0.0851, "mean": 0.0928}
TOL = 5e-5  # 4dp half-width


def load_jsonl(path):
    rows = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            o = json.loads(line)
            rows[o["instanceID"]] = o
    return rows


def f1(tp, fp, fn):
    if tp == 0 and (fp > 0 or fn > 0):
        return 0.0
    if tp == 0 and fp == 0 and fn == 0:
        return None  # class absent from gold AND preds; meaning depends on convention
    return 2 * tp / (2 * tp + fp + fn)


def macro_f1(gold_sets, pred_sets, label_space, labelset_mode, zero_division):
    """gold_sets/pred_sets: instanceID -> set(labels). Missing pred instance -> empty set."""
    per_class = {}
    for c in label_space:
        tp = fp = fn = 0
        for iid, g in gold_sets.items():
            p = pred_sets.get(iid, set())
            if c in g and c in p:
                tp += 1
            elif c in p:
                fp += 1
            elif c in g:
                fn += 1
        per_class[c] = (tp, fp, fn)
    scores = {}
    for c, (tp, fp, fn) in per_class.items():
        v = f1(tp, fp, fn)
        if v is None:  # absent class
            if labelset_mode == "present":
                continue  # excluded from the average entirely
            v = float(zero_division)
        scores[c] = v
    if labelset_mode == "present":
        # average only over classes with gold or predicted support
        pass
    if not scores:
        return 0.0, {}
    return sum(scores.values()) / len(scores), scores


def score(gold_path, preds_path, labelset_mode="fixed", zero_division=0):
    gold = load_jsonl(gold_path)
    preds = load_jsonl(preds_path)
    assert gold, "empty gold file"
    g1 = {i: {r["labels"]["st1"]} for i, r in gold.items()}
    g2 = {i: set(r["labels"]["st2"]) for i, r in gold.items()}
    g3 = {i: set(r["labels"]["st3"]) for i, r in gold.items()}
    p1 = {i: ({r["st1"]} if "st1" in r else set()) for i, r in preds.items()}
    p2 = {i: set(r.get("st2", [])) for i, r in preds.items()}
    p3 = {i: set(r.get("st3", [])) for i, r in preds.items()}
    s1, pc1 = macro_f1(g1, p1, ST1, labelset_mode, zero_division)
    s2, pc2 = macro_f1(g2, p2, ST2, labelset_mode, zero_division)
    s3, pc3 = macro_f1(g3, p3, ST3, labelset_mode, zero_division)
    for name, v in (("st1", s1), ("st2", s2), ("st3", s3)):
        assert 0.0 <= v <= 1.0, f"{name} out of range: {v}"
    mean = (s1 + s2 + s3) / 3
    return {"st1": s1, "st2": s2, "st3": s3, "mean": mean,
            "per_class": {"st1": pc1, "st2": pc2, "st3": pc3},
            "coverage": len(set(preds) & set(gold)) / len(gold)}


def grid(gold_path, preds_path):
    out = {}
    for labelset_mode in ("fixed", "present"):
        for zd in (0, 1):
            r = score(gold_path, preds_path, labelset_mode, zd)
            out[(labelset_mode, zd)] = r
    return out


def main():
    mode, gold_path, preds_path = sys.argv[1], sys.argv[2], sys.argv[3]
    if mode == "pin":
        results = grid(gold_path, preds_path)
        matches = []
        for conv, r in results.items():
            ok = all(abs(r[k] - PIN[k]) <= TOL for k in ("st1", "st2", "st3", "mean"))
            print(f"convention labelset={conv[0]:<8} zero_division={conv[1]}: "
                  f"st1={r['st1']:.6f} st2={r['st2']:.6f} st3={r['st3']:.6f} mean={r['mean']:.6f}"
                  f"  -> {'MATCH' if ok else 'no'}")
            if ok:
                matches.append(conv)
        print(f"\nPIN targets: {PIN} (tol {TOL})")
        if len(matches) == 1:
            print(f"PINNED: exactly one convention matches: {matches[0]}")
            sys.exit(0)
        elif matches:
            print(f"UNPINNED TIE: {matches} all match; default sklearn-like = ('fixed', 0). "
                  f"Record the tie; discriminate via a dev probe / platform precision.")
            sys.exit(0)
        else:
            print("NO MATCH: convention grid exhausted; a dev smoke submission must arbitrate.")
            sys.exit(2)
    elif mode == "score":
        conv = ("fixed", 0)
        if "--convention" in sys.argv:
            name = sys.argv[sys.argv.index("--convention") + 1]
            ls, zd = name.split(",")
            conv = (ls, int(zd))
        r = score(gold_path, preds_path, *conv)
        # determinism self-check
        r2 = score(gold_path, preds_path, *conv)
        assert all(abs(r[k] - r2[k]) == 0 for k in ("st1", "st2", "st3", "mean")), "nondeterministic scorer"
        print(json.dumps({k: round(v, 6) for k, v in r.items() if k != "per_class"} |
                         {"per_class": {t: {c: round(v, 4) for c, v in d.items()}
                                        for t, d in r["per_class"].items()}}, indent=1))
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
