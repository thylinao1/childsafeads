"""Advertiser recurrence across the shared task's channel-disjoint splits.

The splits are disjoint by channelID: no channel in the evaluation split appears in
train or dev. They are NOT disjoint by advertiser. The same brand is filmed by many
creators, so a product page domain that carries gold labels in the labelled pool can
reappear, labelled by the same annotators, in a split a system is scored on.

This script measures the size of that channel and what it is worth, three ways:

  1. Overlap. How many channels and how many advertisers are shared between splits.
  2. A memorisation baseline. Predict a held-out instance's ST1 and ST2 by copying the
     majority label of the same advertiser in the labelled pool, with no model at all.
     Measured train -> dev, which is fully honest: dev gold is public and the pool is
     the training split only.
  3. Leakage inside this project's own validation design. The out-of-fold split is
     GroupKFold by channel, so the same recurrence exists between folds. Rows whose
     advertiser appears in another fold are compared with rows whose advertiser does not.

Advertiser key: the host of product_page.resolved_url, falling back to raw_url, lowercased
with a leading "www." removed. Link shorteners are avoided by preferring the resolved URL.

Usage: python3 clinic/advertiser_overlap.py
"""
import json
import os
import sys
from collections import Counter, defaultdict
from urllib.parse import urlparse

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "eval"))
from local_scorer import ST1, ST2, ST3  # noqa: E402

MULTI_TLD = {"co.uk", "com.au", "co.jp", "com.br", "co.nz", "co.in", "com.mx", "co.za"}


def host_of(row):
    pp = row.get("product_page") or {}
    for key in ("resolved_url", "raw_url"):
        u = (pp.get(key) or "").strip()
        if not u:
            continue
        try:
            h = urlparse(u if "://" in u else "http://" + u).netloc.lower()
        except ValueError:
            continue
        if not h:
            continue
        h = h.split(":")[0]
        if h.startswith("www."):
            h = h[4:]
        parts = h.split(".")
        if len(parts) >= 3 and ".".join(parts[-2:]) in MULTI_TLD:
            return ".".join(parts[-3:])
        if len(parts) >= 2:
            return ".".join(parts[-2:])
        return h
    return None


def load(split):
    return [json.loads(l) for l in open(os.path.join(ROOT, "data", f"{split}.jsonl")) if l.strip()]


def macro_present(gold, pred, space):
    scores = []
    for c in space:
        tp = fp = fn = 0
        for g, p in zip(gold, pred):
            gin, pin = c in g, c in p
            if gin and pin: tp += 1
            elif pin: fp += 1
            elif gin: fn += 1
        if tp == fp == fn == 0:
            continue
        scores.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def main():
    tr, dv, te = load("train"), load("dev"), load("test")
    rows = {"train": tr, "dev": dv, "test": te}

    print("=" * 78)
    print("1. OVERLAP BETWEEN SPLITS")
    print("=" * 78)
    ch = {k: {r["channel_context"]["channelID"] for r in v} for k, v in rows.items()}
    ad = {k: {h for h in (host_of(r) for r in v) if h} for k, v in rows.items()}
    have = {k: sum(1 for r in v if host_of(r)) for k, v in rows.items()}
    for k, v in rows.items():
        print(f"  {k:<6} {len(v):>5} rows  {len(ch[k]):>4} channels  "
              f"{len(ad[k]):>4} advertisers  ({have[k]}/{len(v)} rows carry a resolvable advertiser)")
    pool_ch, pool_ad = ch["train"] | ch["dev"], ad["train"] | ad["dev"]
    print()
    for name, target in (("dev  vs train", "dev"), ("test vs train+dev", "test")):
        base_ch = ch["train"] if target == "dev" else pool_ch
        base_ad = ad["train"] if target == "dev" else pool_ad
        shared_ch = ch[target] & base_ch
        shared_ad = ad[target] & base_ad
        n_rows_seen = sum(1 for r in rows[target] if (host_of(r) or "\0") in base_ad)
        print(f"  {name}:")
        print(f"     channels shared    {len(shared_ch):>4} of {len(ch[target])}  "
              f"({100*len(shared_ch)/len(ch[target]):.1f}%)")
        print(f"     advertisers shared {len(shared_ad):>4} of {len(ad[target])}  "
              f"({100*len(shared_ad)/len(ad[target]):.1f}%)")
        print(f"     ROWS whose advertiser is already labelled: {n_rows_seen} of "
              f"{len(rows[target])} ({100*n_rows_seen/len(rows[target]):.1f}%)")

    print()
    print("=" * 78)
    print("2. MEMORISATION BASELINE  (train -> dev, no model)")
    print("=" * 78)
    st1_by_ad, st2_by_ad = defaultdict(list), defaultdict(list)
    for r in tr:
        h = host_of(r)
        if h:
            st1_by_ad[h].append(r["labels"]["st1"])
            st2_by_ad[h].extend(r["labels"]["st2"])

    seen_idx = [i for i, r in enumerate(dv) if host_of(r) in st1_by_ad]
    unseen_idx = [i for i, r in enumerate(dv) if host_of(r) not in st1_by_ad]
    print(f"  dev rows with an advertiser seen in train: {len(seen_idx)} "
          f"({100*len(seen_idx)/len(dv):.1f}%); unseen: {len(unseen_idx)}")

    g1 = [{r["labels"]["st1"]} for r in dv]
    g2 = [set(r["labels"]["st2"]) for r in dv]
    maj_st1 = Counter(r["labels"]["st1"] for r in tr).most_common(1)[0][0]
    maj_st2 = Counter(l for r in tr for l in r["labels"]["st2"]).most_common(1)[0][0]

    p1_mem, p2_mem, p1_maj, p2_maj = [], [], [], []
    for r in dv:
        h = host_of(r)
        p1_maj.append({maj_st1}); p2_maj.append({maj_st2})
        if h in st1_by_ad:
            p1_mem.append({Counter(st1_by_ad[h]).most_common(1)[0][0]})
            cnt = Counter(st2_by_ad[h]); n = len(st1_by_ad[h])
            keep = {c for c, k in cnt.items() if k / n >= 0.5} or {cnt.most_common(1)[0][0]}
            p2_mem.append(keep)
        else:
            p1_mem.append({maj_st1}); p2_mem.append({maj_st2})

    def sub(lst, idx):
        return [lst[i] for i in idx]

    print()
    print(f"  {'':<34}{'ST1':>9}{'ST2':>9}")
    for label, idx in (("all dev rows", list(range(len(dv)))),
                       ("advertiser-seen rows only", seen_idx),
                       ("advertiser-unseen rows only", unseen_idx)):
        if not idx:
            continue
        a1 = macro_present(sub(g1, idx), sub(p1_mem, idx), ST1)
        a2 = macro_present(sub(g2, idx), sub(p2_mem, idx), ST2)
        b1 = macro_present(sub(g1, idx), sub(p1_maj, idx), ST1)
        b2 = macro_present(sub(g2, idx), sub(p2_maj, idx), ST2)
        print(f"  advertiser-copy  {label:<28}{a1:>9.4f}{a2:>9.4f}")
        print(f"  majority-class   {label:<28}{b1:>9.4f}{b2:>9.4f}")

    exact1 = sum(1 for i in seen_idx if p1_mem[i] == g1[i])
    exact2 = sum(1 for i in seen_idx if p2_mem[i] == g2[i])
    print()
    print(f"  On the {len(seen_idx)} advertiser-seen dev rows, copying the advertiser's "
          f"majority training label is exactly right for")
    print(f"     ST1 {exact1}/{len(seen_idx)} = {100*exact1/max(len(seen_idx),1):.1f}%   "
          f"ST2 (exact set match) {exact2}/{len(seen_idx)} = {100*exact2/max(len(seen_idx),1):.1f}%")

    print()
    print("=" * 78)
    print("3. LEAKAGE INSIDE THIS PROJECT'S OWN CHANNEL-GROUPED CROSS-VALIDATION")
    print("=" * 78)
    folds = json.load(open(os.path.join(ROOT, "eval", "folds.json")))
    fold_of_row = [folds[r["channel_context"]["channelID"]] for r in tr]
    ad_folds = defaultdict(set)
    for r, f in zip(tr, fold_of_row):
        h = host_of(r)
        if h:
            ad_folds[h].add(f)
    cross = [i for i, (r, f) in enumerate(zip(tr, fold_of_row))
             if host_of(r) and len(ad_folds[host_of(r)] - {f}) > 0]
    print(f"  training rows whose advertiser also appears in a DIFFERENT fold: "
          f"{len(cross)} of {len(tr)} ({100*len(cross)/len(tr):.1f}%)")
    print(f"  advertisers spanning more than one fold: "
          f"{sum(1 for a, fs in ad_folds.items() if len(fs) > 1)} of {len(ad_folds)}")
    print()
    print("  Interpretation: GroupKFold by channel does not make the folds advertiser-")
    print("  disjoint, so an out-of-fold estimate is measured partly on advertisers the")
    print("  model has already been fitted on. The evaluation split inherits the same")
    print("  property with respect to the labelled pool.")

    top = Counter()
    for r in te:
        h = host_of(r)
        if h and h in pool_ad:
            top[h] += 1
    print()
    print("  Most frequent already-labelled advertisers in the evaluation split:")
    for h, k in top.most_common(12):
        n_lab = sum(1 for r in tr + dv if host_of(r) == h)
        print(f"     {h:<34} {k:>3} eval rows, {n_lab:>3} labelled rows")


if __name__ == "__main__":
    main()
