"""Shared data/label/metric code for the M1 encoder lane (m1d DeBERTa, m1m ModernBERT)."""
import json
import random

import numpy as np

ST1 = ["physical_goods", "digital_content_or_services", "physical_services", "none", "other"]
ST2 = ["toys", "food", "apps", "hardware_electronics", "fashion", "health", "education",
       "financial", "gambling", "gambling_adjacent", "creator_community", "other"]
ST3 = ["undisclosed_advertising", "inadequate_disclosure", "direct_exhortation",
       "misleading_claim", "age_restricted_or_prohibited_product", "hfss_food_marketing",
       "no_flag", "insufficient_context"]
TASKS = {"st1": ST1, "st2": ST2, "st3": ST3}

# per-field token budgets: (deberta max_len 512, modernbert max_len 2048)
BUDGETS = {
    "deberta":    {"transcript": 290, "title": 22, "description": 90, "channel": 12,
                   "page_title": 14, "page_text": 60, "max_len": 512},
    "modernbert": {"transcript": 1100, "title": 32, "description": 220, "channel": 16,
                   "page_title": 32, "page_text": 560, "max_len": 2048},
}
ARCH_MODEL = {"deberta": "microsoft/deberta-v3-large",
              "modernbert": "answerdotai/ModernBERT-large"}

# non-transcript template fields eligible for train-time field dropout
DROPPABLE = ["title", "description", "disclosure", "channel", "page_title", "page_text"]
FIELD_DROPOUT_P = 0.15


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _trunc(tok, text, budget):
    if not text:
        return ""
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= budget:
        return text
    return tok.decode(ids[:budget], skip_special_tokens=True)


def extract_fields(row, tok, budgets):
    """Per-field token-budget truncation (never naive tail-truncation of the concat)."""
    vc = row.get("video_context") or {}
    cc = row.get("channel_context") or {}
    pp = row.get("product_page") or {}
    disc = str(vc.get("official_disclosure", "") or "").strip().lower()
    disc = disc if disc in ("true", "false") else "unknown"
    return {
        "transcript": _trunc(tok, (row.get("transcript") or {}).get("text", ""), budgets["transcript"]),
        "title": _trunc(tok, vc.get("title", ""), budgets["title"]),
        "description": _trunc(tok, vc.get("description", ""), budgets["description"]),
        "disclosure": disc,
        "channel": _trunc(tok, cc.get("channel_name", ""), budgets["channel"]),
        "page_title": _trunc(tok, pp.get("page_title", ""), budgets["page_title"]),
        "page_text": _trunc(tok, pp.get("text", ""), budgets["page_text"]),
    }


def build_text(fields, dropped=()):
    g = lambda k: "" if k in dropped else fields[k]
    return ("TRANSCRIPT: {t}\nTITLE: {ti}\nDESCRIPTION: {d}\nPAID_PROMOTION_LABEL: {dl}"
            "\nCHANNEL: {c}\nPAGE_TITLE: {pt}\nPRODUCT_PAGE: {p}").format(
        t=fields["transcript"], ti=g("title"), d=g("description"),
        dl=("unknown" if "disclosure" in dropped else fields["disclosure"]),
        c=g("channel"), pt=g("page_title"), p=g("page_text"))


def label_arrays(rows):
    """Return (y1 int [N], y2 float [N,12], y3 float [N,8]); rows must carry labels."""
    n = len(rows)
    y1 = np.zeros(n, dtype=np.int64)
    y2 = np.zeros((n, len(ST2)), dtype=np.float32)
    y3 = np.zeros((n, len(ST3)), dtype=np.float32)
    for i, r in enumerate(rows):
        lab = r["labels"]
        y1[i] = ST1.index(lab["st1"])
        for l in lab["st2"]:
            y2[i, ST2.index(l)] = 1.0
        for l in lab["st3"]:
            y3[i, ST3.index(l)] = 1.0
    return y1, y2, y3


def present_macro_f1(y_true_bin, y_pred_bin):
    """Macro-F1 over classes present in gold-or-pred (both N x K binary arrays)."""
    present = (y_true_bin.sum(0) > 0) | (y_pred_bin.sum(0) > 0)
    f1s = []
    for k in np.where(present)[0]:
        t, p = y_true_bin[:, k], y_pred_bin[:, k]
        tp = float((t * p).sum())
        denom = t.sum() + p.sum()
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def naive_score(y1, y2, y3, p1, p2, p3):
    """Mean of the three present-macro-F1s under naive decisions (argmax / 0.5)."""
    n, k1 = p1.shape
    t1 = np.zeros((n, k1)); t1[np.arange(n), y1] = 1
    q1 = np.zeros((n, k1)); q1[np.arange(n), p1.argmax(1)] = 1
    s1 = present_macro_f1(t1, q1)
    s2 = present_macro_f1(y2 > 0.5, p2 > 0.5)
    s3 = present_macro_f1(y3 > 0.5, p3 > 0.5)
    return (s1 + s2 + s3) / 3.0, (s1, s2, s3)


class FieldDropper:
    """Deterministic per-seed field dropout on non-transcript fields."""
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def sample(self):
        return tuple(f for f in DROPPABLE if self.rng.random() < FIELD_DROPOUT_P)
