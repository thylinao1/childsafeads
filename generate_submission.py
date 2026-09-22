"""Deterministic ChildSafeAds submission generator.

Builds submission.zip = exactly ONE member named 'submission.jsonl' at the zip
root (no directories). Asserts the full contract before writing; refuses to
write a degenerate (near-constant) submission.

Usage: python generate_submission.py --preds preds_dev.jsonl --split dev --out submission_dev.zip

Provenance guard (--from-manifest). Added after the evaluation phase, in which labels set by
hand were placed in files that had been emitted and then edited before packaging. With the
flag, the generator (1) verifies the pinned configuration (clinic/config_manifest.py),
(2) re-emits the split from the pinned member probabilities and decide/params.json, and
(3) refuses to package --preds unless every row is identical to that fresh emission. It then
writes <out>.provenance.json with the archive's sha256, the manifest note and the pinned
member list, so that the file uploaded can be tied back to the configuration that produced it.
"""
import argparse
import hashlib
import json
import os
import sys
import zipfile

ST1 = {"physical_goods", "digital_content_or_services", "physical_services", "none", "other"}
ST2 = {"toys", "food", "apps", "hardware_electronics", "fashion", "health", "education",
       "financial", "gambling", "gambling_adjacent", "creator_community", "other"}
ST3 = {"undisclosed_advertising", "inadequate_disclosure", "direct_exhortation",
       "misleading_claim", "age_restricted_or_prohibited_product", "hfss_food_marketing",
       "no_flag", "insufficient_context"}
ROOT = os.path.dirname(os.path.abspath(__file__))


PARAMS_PATH = os.path.join(ROOT, "decide", "params.json")


def emission_from_manifest(split):
    """Verify the pinned configuration and re-emit `split` from it. Returns
    (instanceID -> canonical row, manifest)."""
    sys.path.insert(0, os.path.join(ROOT, "clinic"))
    sys.path.insert(0, os.path.join(ROOT, "decide"))
    import config_manifest
    import decision_layer
    assert config_manifest.cmd_verify() == 0, "configuration drift: refusing to package"
    manifest = json.load(open(config_manifest.MANIFEST))
    params = json.load(open(PARAMS_PATH))
    assert list(params["members"]) == list(manifest["blend_members"]), \
        f"params.json members {params['members']} differ from the pinned {manifest['blend_members']}"
    rows = decision_layer.emit_preds(manifest["blend_members"], split, params, params.get("weights"))
    fresh = {r["instanceID"]: {"instanceID": r["instanceID"], "st1": r["st1"],
                               "st2": sorted(set(r["st2"])), "st3": sorted(set(r["st3"]))}
             for r in rows}
    return fresh, manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--from-manifest", action="store_true",
                    help="verify the pinned configuration, re-emit the split from it, and refuse "
                         "to package --preds unless it is identical to that emission; write "
                         "<out>.provenance.json")
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

    manifest = None
    if args.from_manifest:
        fresh, manifest = emission_from_manifest(args.split)
        differing = [iid for iid in want_ids if rows[iid] != fresh.get(iid)]
        assert not differing, (
            f"{len(differing)} of {len(want_ids)} rows differ from a fresh emission of the pinned "
            f"configuration (first: {differing[:3]}); refusing to package a file the system did "
            f"not produce")

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
    if manifest is not None:
        prov = {"archive": os.path.basename(args.out),
                "archive_sha256": hashlib.sha256(open(args.out, "rb").read()).hexdigest(),
                "payload_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                "split": args.split, "rows": len(rows),
                "manifest_note": manifest.get("note"),
                "blend_members": manifest["blend_members"],
                "params_md5": hashlib.md5(open(PARAMS_PATH, "rb").read()).hexdigest(),
                "check": "every row identical to a fresh emission of the pinned configuration"}
        with open(args.out + ".provenance.json", "w") as f:
            json.dump(prov, f, indent=1)
        print(f"provenance: {args.out}.provenance.json (archive sha256 {prov['archive_sha256'][:16]}...)")


if __name__ == "__main__":
    main()
