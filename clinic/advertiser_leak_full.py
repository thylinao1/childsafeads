"""Extends clinic/advertiser_overlap.py to all three sub-tasks, to this project's own
system, and to paired uncertainty.

Three questions:
  A. How far does a zero-parameter advertiser lookup get on the FULL metric (mean over
     ST1, ST2, ST3), measured train -> dev?
  B. Does advertiser identity leak COMPLIANCE (ST3) as well as product identity (ST1/ST2)?
  C. Does the submitted system's advantage over that lookup survive on rows whose
     advertiser was never labelled?

Everything is measured on dev, whose gold is public, with the labelled pool restricted to
the training split. The scoring convention is the pinned present-label-set macro-F1; the
majority-class control reproduces the organisers' published baseline row as a check.
"""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.join(ROOT, "decide"))
sys.path.insert(0, os.path.join(ROOT, "clinic"))
from local_scorer import ST1, ST2, ST3  # noqa: E402
from advertiser_overlap import host_of, load, macro_present  # noqa: E402


def build_pool(rows):
    st = {"st1": defaultdict(list), "st2": defaultdict(list), "st3": defaultdict(list)}
    n = defaultdict(int)
    for r in rows:
        h = host_of(r)
        if not h:
            continue
        n[h] += 1
        st["st1"][h].append(r["labels"]["st1"])
        st["st2"][h].extend(r["labels"]["st2"])
        st["st3"][h].extend(r["labels"]["st3"])
    return st, n


def lookup_pred(h, st, n, fallback):
    """Majority single label for ST1; labels present in >=50% of the advertiser's rows for
    the multi-label tasks, never empty."""
    if h not in st["st1"]:
        return dict(fallback)
    out = {"st1": {Counter(st["st1"][h]).most_common(1)[0][0]}}
    for t in ("st2", "st3"):
        c = Counter(st[t][h])
        keep = {k for k, v in c.items() if v / n[h] >= 0.5}
        out[t] = keep or {c.most_common(1)[0][0]}
    return out


def bootstrap_ci(gold, preds, idx_by_channel, reps=4000, seed=0):
    """Channel-cluster bootstrap of a single system's mean macro-F1."""
    chans = sorted(idx_by_channel)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        pick = rng.integers(0, len(chans), size=len(chans))
        idx = [i for k in pick for i in idx_by_channel[chans[k]]]
        m = 0.0
        for t, space in (("st1", ST1), ("st2", ST2), ("st3", ST3)):
            m += macro_present([gold[t][i] for i in idx], [preds[t][i] for i in idx], space)
        vals.append(m / 3)
    v = np.asarray(vals)
    return float(v.mean()), float(v.std(ddof=1)), [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    tr, dv = load("train"), load("dev")
    st, n = build_pool(tr)
    fallback = {"st1": {Counter(r["labels"]["st1"] for r in tr).most_common(1)[0][0]},
                "st2": {Counter(l for r in tr for l in r["labels"]["st2"]).most_common(1)[0][0]},
                "st3": {Counter(l for r in tr for l in r["labels"]["st3"]).most_common(1)[0][0]}}
    print("majority-class fallback:", {k: sorted(v) for k, v in fallback.items()})

    gold = {"st1": [{r["labels"]["st1"]} for r in dv],
            "st2": [set(r["labels"]["st2"]) for r in dv],
            "st3": [set(r["labels"]["st3"]) for r in dv]}
    look = {"st1": [], "st2": [], "st3": []}
    majo = {"st1": [], "st2": [], "st3": []}
    for r in dv:
        p = lookup_pred(host_of(r), st, n, fallback)
        for t in ("st1", "st2", "st3"):
            look[t].append(p[t]); majo[t].append(set(fallback[t]))

    seen = [i for i, r in enumerate(dv) if host_of(r) in st["st1"]]
    unseen = [i for i, r in enumerate(dv) if host_of(r) not in st["st1"]]

    # the submitted system's own dev predictions, if the emitted file is present
    sysp = None
    for cand in ("preds_dev_canonical.jsonl", "preds_dev_blend.jsonl"):
        pth = os.path.join(ROOT, cand)
        if os.path.exists(pth):
            byid = {json.loads(l)["instanceID"]: json.loads(l) for l in open(pth) if l.strip()}
            if all(r["instanceID"] in byid for r in dv):
                sysp = {"st1": [{byid[r["instanceID"]]["st1"]} for r in dv],
                        "st2": [set(byid[r["instanceID"]]["st2"]) for r in dv],
                        "st3": [set(byid[r["instanceID"]]["st3"]) for r in dv]}
                print(f"system dev predictions loaded from {cand}")
            break

    systems = [("advertiser lookup (no model)", look), ("majority class", majo)]
    if sysp:
        systems.append(("submitted ensemble", sysp))

    def sub(d, idx):
        return {t: [d[t][i] for i in idx] for t in d}

    print()
    hdr = f"{'system':<30}{'subset':<20}{'n':>5}{'ST1':>9}{'ST2':>9}{'ST3':>9}{'mean':>9}"
    print(hdr); print("-" * len(hdr))
    table = {}
    for name, P in systems:
        for label, idx in (("all dev", list(range(len(dv)))),
                           ("advertiser-seen", seen), ("advertiser-unseen", unseen)):
            g, p = sub(gold, idx), sub(P, idx)
            s1 = macro_present(g["st1"], p["st1"], ST1)
            s2 = macro_present(g["st2"], p["st2"], ST2)
            s3 = macro_present(g["st3"], p["st3"], ST3)
            table[(name, label)] = (s1, s2, s3, (s1 + s2 + s3) / 3)
            print(f"{name:<30}{label:<20}{len(idx):>5}{s1:>9.4f}{s2:>9.4f}{s3:>9.4f}{(s1+s2+s3)/3:>9.4f}")

    print()
    print("B. Does advertiser identity leak COMPLIANCE as well as product identity?")
    a = table[("advertiser lookup (no model)", "advertiser-seen")]
    b = table[("majority class", "advertiser-seen")]
    for i, t in enumerate(("ST1", "ST2", "ST3")):
        print(f"   {t}: advertiser lookup {a[i]:.4f} vs majority {b[i]:.4f}   "
              f"gain {a[i]-b[i]:+.4f}")

    idx_by_ch = defaultdict(list)
    for i, r in enumerate(dv):
        idx_by_ch[r["channel_context"]["channelID"]].append(i)
    print()
    print("Channel-cluster bootstrap on all dev rows (4000 replicates):")
    for name, P in systems:
        m, se, ci = bootstrap_ci(gold, P, idx_by_ch)
        print(f"   {name:<30} mean {m:.4f}  se {se:.4f}  CI [{ci[0]:.4f}, {ci[1]:.4f}]")

    json.dump({str(k): v for k, v in table.items()},
              open(os.path.join(ROOT, "clinic", "advertiser_leak.json"), "w"), indent=1)
    print("\nwrote clinic/advertiser_leak.json")


if __name__ == "__main__":
    main()
