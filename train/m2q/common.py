"""M2 LoRA lane (member m2q) shared code: labels, input template, data loading.

Input template: full fields, 2048-token budget. Transcript first (most important),
product-page text last so tail truncation eats the page text, not the transcript.
"""
import json
import os

ST1 = ["physical_goods", "digital_content_or_services", "physical_services", "none", "other"]
ST2 = ["toys", "food", "apps", "hardware_electronics", "fashion", "health", "education",
       "financial", "gambling", "gambling_adjacent", "creator_community", "other"]
ST3 = ["undisclosed_advertising", "inadequate_disclosure", "direct_exhortation",
       "misleading_claim", "age_restricted_or_prohibited_product", "hfss_food_marketing",
       "no_flag", "insufficient_context"]

LABELS = {"st1": ST1, "st2": ST2, "st3": ST3}
SEED = 42
MAX_LEN = 2048

# char caps before tokenization so mid-priority fields cannot starve the page text
DESC_CHAR_CAP = 1500
PAGE_CHAR_CAP = 6000


def build_input(row):
    t = row.get("transcript") or {}
    v = row.get("video_context") or {}
    c = row.get("channel_context") or {}
    p = row.get("product_page") or {}
    od = v.get("official_disclosure")
    od_s = "unknown" if od in (None, "") else str(od)
    desc = (v.get("description") or "")[:DESC_CHAR_CAP]
    page = (p.get("text") or "")[:PAGE_CHAR_CAP]
    parts = [
        "Sponsored segment from a YouTube video. Classify the advertised product and child-safety flags.",
        f"TRANSCRIPT SEGMENT ({t.get('segment_start','?')}s-{t.get('segment_end','?')}s):",
        (t.get("text") or "").strip(),
        f"VIDEO TITLE: {v.get('title') or ''}",
        f"OFFICIAL DISCLOSURE FLAG: {od_s}",
        f"CHANNEL: {c.get('channel_name') or ''}",
        f"VIDEO DESCRIPTION: {desc}",
        f"PRODUCT PAGE URL: {p.get('resolved_url') or p.get('raw_url') or ''}",
        f"PRODUCT PAGE TITLE: {p.get('page_title') or ''}",
        "PRODUCT PAGE TEXT:",
        page,
    ]
    return "\n".join(parts)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def load_data(repo):
    train = load_jsonl(os.path.join(repo, "data", "train.jsonl"))
    dev = load_jsonl(os.path.join(repo, "data", "dev.jsonl"))
    with open(os.path.join(repo, "eval", "folds.json")) as f:
        folds = json.load(f)
    for r in train:
        r["fold"] = folds[r["channel_context"]["channelID"]]
    return train, dev


def load_test(repo):
    path = os.path.join(repo, "data", "test.jsonl")
    if not os.path.exists(path):
        return None
    return load_jsonl(path)


def targets(row, task):
    labels = LABELS[task]
    y = row["labels"][task]
    if task == "st1":
        return labels.index(y)
    vec = [0.0] * len(labels)
    for l in y:
        vec[labels.index(l)] = 1.0
    return vec
