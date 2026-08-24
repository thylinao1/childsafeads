"""Recover every hand-set label by diffing the submitted archives against each other.

The working notes are not a reliable record of which labels were set by hand: they
undercounted the edits and misattributed the basis of one. The archives are reliable, because
each is the exact file that was uploaded. This script reconstructs the record from them.

It answers three questions:
  1. Which fields differ between each submitted entry and the model-only entry (slot 3)?
  2. Which of the archives is the one that was submitted as slot 4? The platform did not
     timestamp them individually, so the identification is arithmetic: the returned ST1 column
     for slot 4 (0.7378) is consistent with two hand-set `other` labels and not with one.
  3. For each edited instance, what precedent, if any, exists in the labelled data?

The archives themselves are not distributed with this repository.

Usage: python3 clinic/hand_label_audit.py [--downloads <dir with the submitted zips>]
"""
import argparse
import json
import os
import re
import zipfile
from collections import Counter

ARCHIVES = [
    ("slot 1", "childsafeads-sub1-2026-08-11.zip"),
    ("slot 2", "childsafeads-sub2-2026-08-12.zip"),
    ("slot 3 (model only)", "childsafeads-sub3-2026-08-12.zip"),
    ("slot 3 variant", "childsafeads-sub4-2026-08-12.zip"),
    ("slot 4 candidate, 1 label", "childsafeads-sub4b-omaze.zip"),
    ("slot 4 candidate, 2 labels", "childsafeads-sub4c-2rows.zip"),
    ("unsubmitted, 2 labels no ST3", "childsafeads-sub4d-sub3base-2rows.zip"),
    ("unsubmitted, 1 label only", "childsafeads-sub5-carthrottle-only.zip"),
    ("slot 5 (FINAL, submitted)", "childsafeads-FINAL-sub5b-ct-plus-gambling2.zip"),
]
BASE = "childsafeads-sub3-2026-08-12.zip"

# returned leaderboard columns, from METHOD.md section 6.2
RETURNED = {"slot 3 (model only)": 0.6021, "slot 4": 0.7378, "slot 5 (FINAL, submitted)": 0.8049}


def load(path):
    with zipfile.ZipFile(path) as z:
        name = [n for n in z.namelist() if n.endswith(".jsonl")][0]
        return {json.loads(l)["instanceID"]: json.loads(l)
                for l in z.read(name).decode().splitlines() if l.strip()}


def norm(v):
    return sorted(v) if isinstance(v, list) else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--downloads", default=os.path.expanduser("~/Downloads"))
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
    args = ap.parse_args()

    base_path = os.path.join(args.downloads, BASE)
    if not os.path.exists(base_path):
        raise SystemExit(f"base archive not found: {base_path}")
    base = load(base_path)

    print("=" * 92)
    print("1. EVERY ARCHIVE, DIFFED AGAINST THE MODEL-ONLY ENTRY (slot 3)")
    print("=" * 92)
    print(f"{'archive':<32}{'ST1 rows':>10}{'ST2 rows':>10}{'ST3 rows':>10}   ST1 class counts")
    edits = {}
    for label, fn in ARCHIVES:
        p = os.path.join(args.downloads, fn)
        if not os.path.exists(p):
            print(f"{label:<32}  MISSING ({fn})")
            continue
        cur = load(p)
        d = {t: [i for i in base if norm(base[i][t]) != norm(cur[i][t])] for t in ("st1", "st2", "st3")}
        edits[label] = (cur, d)
        c = Counter(r["st1"] for r in cur.values())
        print(f"{label:<32}{len(d['st1']):>10}{len(d['st2']):>10}{len(d['st3']):>10}   "
              f"other={c.get('other', 0)} pg={c.get('physical_goods', 0)} none={c.get('none', 0)}")

    print()
    print("=" * 92)
    print("2. WHICH ARCHIVE WAS SUBMITTED AS SLOT 4")
    print("=" * 92)
    print("Both candidates differ from slot 3 on 53 ST3 rows and on no ST2 row, matching the")
    print("slot 4 record. They are separated only by the returned ST1 column.")
    print()
    print("  A vector carrying only the correct `other` label returns ST1 = 0.8049 (slot 5 did).")
    print("  Slot 4 returned ST1 = 0.7378, a deficit of 0.0671.")
    print("  A second, incorrect `other` takes that class from F1 = 2*1/(2*1+0+0) = 1.000")
    print("  to F1 = 2*1/(2*1+1+0) = 0.667. Spread over the 5 present ST1 classes that is")
    print(f"  {(1 - 2/3)/5:.4f}, with the remainder from the lost `physical_goods` true positive.")
    print("  => slot 4 is the TWO-label archive. Four hand edits were submitted in total.")

    print()
    print("=" * 92)
    print("3. PRECEDENT IN THE LABELLED DATA FOR EACH EDITED INSTANCE")
    print("=" * 92)
    labelled = []
    for split in ("train", "dev"):
        with open(os.path.join(args.data, f"{split}.jsonl")) as f:
            labelled += [json.loads(l) for l in f if l.strip()]

    final = edits.get("slot 5 (FINAL, submitted)", (None, None))[0]
    slot4 = edits.get("slot 4 candidate, 2 labels", (None, None))[0]
    targets = []
    if final:
        for t in ("st1", "st2", "st3"):
            for i in [i for i in base if norm(base[i][t]) != norm(final[i][t])]:
                targets.append(("slot 5", i, t, base[i][t], final[i][t]))
    if slot4:
        for i in [i for i in base if base[i]["st1"] != slot4[i]["st1"]]:
            if not any(x[1] == i for x in targets):
                targets.append(("slot 4 only", i, "st1", base[i]["st1"], slot4[i]["st1"]))

    test = {json.loads(l)["instanceID"]: json.loads(l)
            for l in open(os.path.join(args.data, "test.jsonl")) if l.strip()}
    BRAND = re.compile(r"[a-z0-9]{5,}")
    for where, iid, field, old, new in targets:
        row = test[iid]
        tr = (row.get("transcript") or {}).get("text") or ""
        ch = (row.get("channel_context") or {}).get("channel_name")
        print(f"\n  [{where}] {iid}")
        print(f"     channel {ch} | {field}: {old} -> {new}")
        # find which labelled brand tokens occur in this transcript
        found = []
        for brand in {"omaze", "prizepick", "prize pick", "draftkings", "picklebet", "establishedtitles",
                      "established titles"}:
            if re.search(brand.replace(" ", r"\s?"), tr, re.I):
                found.append(brand)
        if not found:
            url = ((row.get("product_page") or {}).get("resolved_url") or "")
            found = [w for w in BRAND.findall(url.lower()) if w not in
                     ("https", "http", "www", "com", "utm", "source", "medium", "campaign")][:1]
        for b in sorted(set(found)):
            rx = re.compile(b.replace(" ", r"\s?"), re.I)
            hits = [r for r in labelled
                    if rx.search(((r.get("transcript") or {}).get("text") or "") + " " +
                                 ((r.get("product_page") or {}).get("resolved_url") or ""))]
            st1c = Counter(r["labels"]["st1"] for r in hits)
            st2c = Counter(l for r in hits for l in r["labels"]["st2"])
            verdict = ("NO PRECEDENT" if not hits else
                       f"ST1 {dict(st1c)} | ST2 {dict(st2c)}")
            print(f"     brand '{b}': {len(hits)} labelled segments -> {verdict}")


if __name__ == "__main__":
    main()
