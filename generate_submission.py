"""Deterministic ChildSafeAds submission generator.

Builds submission.zip = exactly ONE member named 'submission.jsonl' at the zip
root (no directories). Asserts the full contract before writing; refuses to
write a degenerate (near-constant) submission.

Usage: python generate_submission.py --preds preds_dev.jsonl --split dev --out submission_dev.zip
"""
import argparse
import json
import os
import zipfile

ST1 = {"physical_goods", "digital_content_or_services", "physical_services", "none", "other"}
ST2 = {"toys", "food", "apps", "hardware_electronics", "fashion", "health", "education",
       "financial", "gambling", "gambling_adjacent", "creator_community", "other"}
ST3 = {"undisclosed_advertising", "inadequate_disclosure", "direct_exhortation",
       "misleading_claim", "age_restricted_or_prohibited_product", "hfss_food_marketing",
       "no_flag", "insufficient_context"}
ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    split_file = os.path.join(ROOT, "data", f"{args.split}.jsonl")
    want_ids = [json.loads(l)["instanceID"] for l in open(split_file) if l.strip()]
    want_set = set(want_ids)
    assert len(want_ids) == len(want_set), "duplicate ids in split file?!"

    rows = {}
    for ln, line in enumerate(open(args.preds), 1):
        if not line.strip():
            continue
        o = json.loads(line)
        iid = o["instanceID"]
        assert iid not in rows, f"duplicate prediction for {iid}"
        assert iid in want_set, f"line {ln}: unknown instanceID {iid}"
        assert isinstance(o["st1"], str) and o["st1"] in ST1, f"line {ln}: bad st1 {o.get('st1')!r}"
        assert isinstance(o["st2"], list) and o["st2"] and set(o["st2"]) <= ST2 and \
            len(set(o["st2"])) == len(o["st2"]), f"line {ln}: bad st2 {o.get('st2')!r}"
        s3 = o["st3"]
        assert isinstance(s3, list) and s3 and set(s3) <= ST3 and len(set(s3)) == len(s3), \
            f"line {ln}: bad st3 {s3!r}"
        s3s = set(s3)
        for solo in ("no_flag", "insufficient_context"):
            assert solo not in s3s or len(s3s) == 1, f"line {ln}: {solo} must stand alone"
        assert not ({"undisclosed_advertising", "inadequate_disclosure"} <= s3s), \
            f"line {ln}: T1.1 and T1.2 are mutually exclusive"
        rows[iid] = {"instanceID": iid, "st1": o["st1"], "st2": sorted(set(o["st2"])),
                     "st3": sorted(s3s)}
    missing = want_set - set(rows)
    assert not missing, f"{len(missing)} split instances missing predictions (they'd score 0)"

    # degenerate-output guard: refuse a near-constant submission (silent fallback to one class)
    st1_mode = max(set(r["st1"] for r in rows.values()),
                   key=lambda c: sum(r["st1"] == c for r in rows.values()))
    st1_frac = sum(r["st1"] == st1_mode for r in rows.values()) / len(rows)
    st3_mode = max(set(tuple(r["st3"]) for r in rows.values()),
                   key=lambda c: sum(tuple(r["st3"]) == c for r in rows.values()))
    st3_frac = sum(tuple(r["st3"]) == st3_mode for r in rows.values()) / len(rows)
    assert st1_frac < 0.99 and st3_frac < 0.99, \
        f"degenerate output: st1 mode {st1_frac:.2%}, st3 mode {st3_frac:.2%}"

    payload = "".join(json.dumps(rows[i]) + "\n" for i in want_ids)  # split order, deterministic
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        info = zipfile.ZipInfo("submission.jsonl", date_time=(2026, 1, 1, 0, 0, 0))
        zf.writestr(info, payload)
    with zipfile.ZipFile(args.out) as zf:
        names = zf.namelist()
        assert names == ["submission.jsonl"], f"zip structure wrong: {names}"
    print(f"wrote {args.out}: {len(rows)} rows, st1-mode {st1_frac:.1%}, zip=[submission.jsonl]")


if __name__ == "__main__":
    main()
