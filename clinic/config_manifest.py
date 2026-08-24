"""Pin the exact configuration a validated score belongs to, and refuse to build without it.

Why this exists (2026-08-12). Two near-misses, both of the same shape: every
individual file valid, the SET wrong.
  1. a packaging step rsynced a regenerated 4-seed member merge over the validated 3-seed one and built a
     checker-green artifact for a config never evaluated (and measurably worse).
  2. The s3s stacker's MEMBERS list is edited by hand when testing new inputs. A leftover or
     missing entry silently changes the feature set the stacker was validated with.
A content check catches (1). Only a manifest catches (2), because nothing is corrupt, the
composition is just different from the one that earned the number.

  python clinic/config_manifest.py record "OOF 0.6920"   # after a config's score is validated
  python clinic/config_manifest.py verify                # before building anything shippable

verify exits 1 on any drift: blend membership, stacker inputs, or the bytes of any npz the
config consumes. Both directions, missing and extra.
"""
import hashlib
import json
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MANIFEST = os.path.join(ROOT, "CONFIG-MANIFEST.json")
TASKS = ("st1", "st2", "st3")
SPLITS = ("oof", "dev", "test")


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def s3s_members():
    src = open(os.path.join(ROOT, "train", "s3s", "stack_st3.py")).read()
    m = re.search(r"^MEMBERS = \[(.*?)\]", src, re.M | re.S)
    assert m, "could not find MEMBERS in stack_st3.py"
    return re.findall(r'"([^"]+)"', m.group(1))


def current():
    params = json.load(open(os.path.join(ROOT, "decide", "params.json")))
    blend = list(params["members"])
    stack = s3s_members()
    # every npz this configuration actually reads
    consumed = set()
    for m in blend + stack:
        for t in TASKS:
            for sp in SPLITS:
                p = os.path.join(ROOT, "preds", f"{m}_{t}_{sp}.npz")
                if os.path.exists(p):
                    consumed.add(f"{m}_{t}_{sp}.npz")
    # decision-layer constants live in code, not params.json, and a silent edit to them
    # changes the shipped system while leaving every file hash intact. 2026-08-12: an
    # over-eager revert emptied ST3_COL_ONLY and the manifest passed anyway, costing
    # st3 0.6199 -> 0.6132 with no warning. Pin them.
    dsrc = open(os.path.join(ROOT, "decide", "decision_layer.py")).read()
    consts = {}
    for name in ("ST3_COL_EXTRA", "ST3_COL_ONLY", "NEVER_PREDICT_ST1", "SPONSOR_RX",
                 "DESC_DISCLOSURE_RX", "IC_PREEMPT_EXEMPT"):
        m = re.search(rf"^{name} = (.+?)$", dsrc, re.M)
        consts[name] = m.group(1).strip() if m else "ABSENT"
    return {
        "blend_members": blend,
        "s3s_members": stack,
        "decision_constants": consts,
        "st3_nf": params.get("st3_nf"),
        "st1_other_tau": params.get("st1_other_tau"),
        "files": {f: md5(os.path.join(ROOT, "preds", f)) for f in sorted(consumed)},
    }


def cmd_record(note):
    cur = current()
    cur["note"] = note
    json.dump(cur, open(MANIFEST, "w"), indent=1)
    print(f"recorded: blend={cur['blend_members']}")
    print(f"          s3s={cur['s3s_members']}")
    print(f"          {len(cur['files'])} npz pinned by md5")
    print(f"          note: {note}")


def cmd_verify():
    if not os.path.exists(MANIFEST):
        print("FATAL: no CONFIG-MANIFEST.json. Record one after the config's score is validated.")
        return 1
    want = json.load(open(MANIFEST))
    cur = current()
    bad = []
    if cur["blend_members"] != want["blend_members"]:
        bad.append(f"blend members: have {cur['blend_members']}, pinned {want['blend_members']}")
    if cur["s3s_members"] != want["s3s_members"]:
        bad.append(f"s3s members: have {cur['s3s_members']}, pinned {want['s3s_members']}")
    for k in ("st3_nf", "st1_other_tau"):
        if cur[k] != want.get(k):
            bad.append(f"{k}: have {cur[k]}, pinned {want.get(k)}")
    for name, val in cur.get("decision_constants", {}).items():
        pinned = (want.get("decision_constants") or {}).get(name, "NOT-PINNED")
        if pinned != "NOT-PINNED" and val != pinned:
            bad.append(f"{name} changed:\n      have   {val}\n      pinned {pinned}")
    missing = sorted(set(want["files"]) - set(cur["files"]))
    extra = sorted(set(cur["files"]) - set(want["files"]))
    if missing:
        bad.append(f"{len(missing)} pinned npz MISSING: {missing[:4]}")
    if extra:
        bad.append(f"{len(extra)} npz present that the pinned config does not use: {extra[:4]}")
    changed = [f for f in sorted(set(want["files"]) & set(cur["files"]))
               if want["files"][f] != cur["files"][f]]
    if changed:
        bad.append(f"{len(changed)} npz CHANGED since the manifest was recorded: {changed[:4]}")
    if bad:
        print("CONFIG DRIFT, refusing to vouch for this build:")
        for b in bad:
            print("  " + b)
        print(f"\nPinned config was: {want.get('note')}")
        return 1
    print(f"config matches the pinned manifest ({want.get('note')}); "
          f"{len(cur['files'])} npz, blend {cur['blend_members']}")
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if mode == "record":
        cmd_record(sys.argv[2] if len(sys.argv) > 2 else "unlabelled")
        sys.exit(0)
    sys.exit(cmd_verify())
