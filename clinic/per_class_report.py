"""Per-class precision, recall and F1 for each sub-task, plus the ST1 confusion matrix,
on the two labelled splits (out-of-fold over train, and dev), under the pinned
present-label-set convention of eval/local_scorer.py.

Usage:
    python clinic/per_class_report.py --oof-preds preds_oof.jsonl --dev-preds preds_dev.jsonl
        [--json clinic/per_class_report.json] [--latex]

The prediction files are produced by decide/decision_layer.py emit with the canonical
member list (see README). The script is a pure function of the two prediction files and
the two gold files; it fits nothing.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from eval.local_scorer import ST1, ST2, ST3, load_jsonl  # noqa: E402

SPACES = {"st1": ST1, "st2": ST2, "st3": ST3}


def sets_for(gold, preds, task):
    if task == "st1":
        g = {i: {r["labels"]["st1"]} for i, r in gold.items()}
        p = {i: ({r["st1"]} if "st1" in r else set()) for i, r in preds.items()}
    else:
        g = {i: set(r["labels"][task]) for i, r in gold.items()}
        p = {i: set(r.get(task, [])) for i, r in preds.items()}
    return g, p


def per_class(gold, preds, task):
    g, p = sets_for(gold, preds, task)
    out = {}
    for c in SPACES[task]:
        tp = fp = fn = 0
        for iid, gs in g.items():
            ps = p.get(iid, set())
            if c in gs and c in ps:
                tp += 1
            elif c in ps:
                fp += 1
            elif c in gs:
                fn += 1
        support, predicted = tp + fn, tp + fp
        prec = tp / predicted if predicted else None
        rec = tp / support if support else None
        if support == 0 and predicted == 0:
            f1 = None  # absent from gold and predictions: dropped by the present convention
        elif tp == 0:
            f1 = 0.0
        else:
            f1 = 2 * tp / (2 * tp + fp + fn)
        out[c] = {"support": support, "predicted": predicted, "tp": tp, "fp": fp, "fn": fn,
                  "precision": prec, "recall": rec, "f1": f1}
    present = [v["f1"] for v in out.values() if v["f1"] is not None]
    macro = sum(present) / len(present) if present else 0.0
    return out, macro


def confusion_st1(gold, preds):
    m = {g: {p: 0 for p in ST1} for g in ST1}
    for iid, r in gold.items():
        g = r["labels"]["st1"]
        p = preds.get(iid, {}).get("st1")
        if p is not None:
            m[g][p] += 1
    return m


def build(oof_preds, dev_preds):
    gold = {"oof": load_jsonl(os.path.join(ROOT, "data", "train.jsonl")),
            "dev": load_jsonl(os.path.join(ROOT, "data", "dev.jsonl"))}
    preds = {"oof": load_jsonl(oof_preds), "dev": load_jsonl(dev_preds)}
    report = {}
    for split in ("oof", "dev"):
        assert set(preds[split]) == set(gold[split]), f"{split}: prediction ids differ from gold"
        report[split] = {"n": len(gold[split]), "tasks": {}, "macro": {}}
        for task in ("st1", "st2", "st3"):
            pc, macro = per_class(gold[split], preds[split], task)
            report[split]["tasks"][task] = pc
            report[split]["macro"][task] = macro
        report[split]["macro"]["mean"] = sum(report[split]["macro"][t] for t in ("st1", "st2", "st3")) / 3
        report[split]["confusion_st1"] = confusion_st1(gold[split], preds[split])
    return report


SHORT = {
    "physical_goods": "physical goods", "digital_content_or_services": "digital content/services",
    "physical_services": "physical services", "none": "none", "other": "other",
    "toys": "toys", "food": "food", "apps": "apps", "hardware_electronics": "hardware/electronics",
    "fashion": "fashion", "health": "health", "education": "education", "financial": "financial",
    "gambling": "gambling", "gambling_adjacent": "gambling-adjacent",
    "creator_community": "creator community",
    "undisclosed_advertising": "undisclosed advertising",
    "inadequate_disclosure": "inadequate disclosure", "direct_exhortation": "direct exhortation",
    "misleading_claim": "misleading claim",
    "age_restricted_or_prohibited_product": "age-restricted product",
    "hfss_food_marketing": "HFSS food marketing", "no_flag": "no flag",
    "insufficient_context": "insufficient context",
}


def fmt(v):
    return "--" if v is None else f"{v:.2f}"


def latex(report):
    """One compact table: per class, gold support and F1 on OOF and dev, with precision and
    recall on OOF (the larger split)."""
    lines = []
    lines.append(r"\begin{tabular}{@{}lrrrrrr@{}}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{4}{c}{out-of-fold (2353)} & \multicolumn{2}{c}{dev (504)} \\")
    lines.append(r"\cmidrule(lr){2-5}\cmidrule(l){6-7}")
    lines.append(r"Class & $n$ & P & R & F1 & $n$ & F1 \\")
    for task, title in (("st1", "ST1"), ("st2", "ST2"), ("st3", "ST3")):
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\emph{{{title}, macro-F1 "
                     rf"{report['oof']['macro'][task]:.4f} / {report['dev']['macro'][task]:.4f}}}}} \\")
        for c in SPACES[task]:
            o = report["oof"]["tasks"][task][c]
            d = report["dev"]["tasks"][task][c]
            lines.append(f"{SHORT[c]} & {o['support']} & {fmt(o['precision'])} & {fmt(o['recall'])} & "
                         f"{fmt(o['f1'])} & {d['support']} & {fmt(d['f1'])} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def latex_confusion(report, split):
    m = report[split]["confusion_st1"]
    ab = {"physical_goods": "PG", "digital_content_or_services": "DC", "physical_services": "PS",
          "none": "N", "other": "O"}
    lines = [r"\begin{tabular}{@{}l" + "r" * len(ST1) + "@{}}", r"\toprule",
             "gold $\\backslash$ pred & " + " & ".join(ab[c] for c in ST1) + r" \\", r"\midrule"]
    for g in ST1:
        lines.append(f"{ab[g]} ({SHORT[g]}) & " + " & ".join(str(m[g][p]) for p in ST1) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-preds", required=True)
    ap.add_argument("--dev-preds", required=True)
    ap.add_argument("--json", default=os.path.join(ROOT, "clinic", "per_class_report.json"))
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()
    report = build(args.oof_preds, args.dev_preds)
    with open(args.json, "w") as f:
        json.dump(report, f, indent=1)
    for split in ("oof", "dev"):
        m = report[split]["macro"]
        print(f"{split}: st1 {m['st1']:.4f} st2 {m['st2']:.4f} st3 {m['st3']:.4f} mean {m['mean']:.4f}")
        for task in ("st1", "st2", "st3"):
            for c, v in report[split]["tasks"][task].items():
                print(f"  {task} {c:38s} n={v['support']:4d} pred={v['predicted']:4d} "
                      f"P={fmt(v['precision'])} R={fmt(v['recall'])} F1={fmt(v['f1'])}")
        print("  ST1 confusion (gold rows, pred cols, order " + ", ".join(ST1) + "):")
        for g in ST1:
            print("   ", f"{g:30s}", [report[split]["confusion_st1"][g][p] for p in ST1])
    if args.latex:
        print("\n% ---- per-class table ----")
        print(latex(report))
        print("\n% ---- ST1 confusion, OOF ----")
        print(latex_confusion(report, "oof"))
        print("\n% ---- ST1 confusion, dev ----")
        print(latex_confusion(report, "dev"))
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
