"""Pipeline figure for the system description paper. Writes paper/latex/fig/pipeline.pdf.

Pure drawing: every architectural fact shown here restates METHOD.md sections 3-5
(members, equal-weight blend, ST3 column adjustments, decision layer). No data is read.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "paper", "latex", "fig")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.size": 9.5, "font.family": "serif",
                     "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
                     "pdf.fonttype": 42, "ps.fonttype": 42})

C_ST = {"ST1": "#4c72b0", "ST2": "#55a868", "ST3": "#c44e52"}
C_ENC, C_LLM, C_CLS, C_EDGE, C_HEAD = "#edf1f7", "#faf3dd", "#f2f2f2", "#444444", "#7a7a7a"
W_IN, H_IN = 6.4, 2.85

fig, ax = plt.subplots(figsize=(W_IN, H_IN))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(x, y, w, h, fc, lw=0.7, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=fc, ec=C_EDGE, lw=lw, ls=ls, mutation_aspect=W_IN / H_IN))


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6,
                                 lw=0.7, color=C_EDGE, shrinkA=0, shrinkB=0))


def chip(x, y, text, fc, tc="white", fs=5.6, ha="center"):
    ax.text(x, y, text, fontsize=fs, color=tc, ha=ha, va="center",
            bbox=dict(boxstyle="round,pad=0.22", fc=fc, ec="none"))


def header(x, text):
    ax.text(x, 97.2, text, fontsize=5.6, color=C_HEAD, ha="center", va="center")


# ---- column 1: instance fields, one row per data access level ----
X_IN, W_INBOX = 0.5, 20.0
box(X_IN, 10, W_INBOX, 80, "white")
header(X_IN + W_INBOX / 2, "INSTANCE FIELDS")
ax.plot([X_IN + 1.6, X_IN + W_INBOX - 1.6], [83.5, 83.5], color="#cccccc", lw=0.5)
ax.text(X_IN + W_INBOX / 2, 87.6, "one segment", fontsize=6.4, ha="center", va="center",
        style="italic")
rows = ((74.5, "L1", "transcript"),
        (61.5, "L2", "+ title, description,\n   paid-promotion label"),
        (46.0, "L3", "+ channel name"),
        (33.0, "L4", "+ product-page\n   title and text"))
for yy, lvl, txt in rows:
    chip(X_IN + 3.0, yy, lvl, "#e4e4e4", tc="#444444", fs=5.4)
    ax.text(X_IN + 6.2, yy, txt, fontsize=6.0, ha="left", va="center", linespacing=1.3)
ax.text(X_IN + W_INBOX / 2, 17, "each level adds\nthe fields above it", fontsize=5.5,
        ha="center", va="center", color=C_HEAD, style="italic", linespacing=1.25)

# ---- column 2: six members ----
members = [
    ("m1d", "DeBERTa-v3-large, fine-tuned,\n3 seeds x 5 channel folds", C_ENC,
     ("ST1", "ST2", "ST3")),
    ("m1m", "ModernBERT-large, fine-tuned,\n3 seeds x 5 channel folds", C_ENC,
     ("ST1", "ST2", "ST3")),
    ("m3q32A", "Qwen3-32B, zero-shot, guided\nJSON, logprob probabilities", C_LLM,
     ("ST1", "ST2", "ST3")),
    ("m0s", "field-aware TF-IDF +\nlogistic regression", C_CLS, ("ST1", "ST2", "ST3")),
    ("m2qS2", "Qwen3-14B LoRA classifier;\nST1, ST3 held neutral", C_ENC, ("ST2",)),
    ("s3s", "logistic stacker over member\nST3 + aux features; ST1, ST2 neutral", C_CLS,
     ("ST3",)),
]
MX, MW, MH, GAP = 24.5, 22.5, 13.3, 1.5
header(MX + MW / 2, "ENSEMBLE MEMBERS")
TOP = 91.5
for i, (name, desc, fc, ts) in enumerate(members):
    top = TOP - i * (MH + GAP)
    my = top - MH
    box(MX, my, MW, MH, fc)
    ax.text(MX + 1.2, top - 3.2, name, fontsize=6.8, ha="left", va="center",
            family="monospace", fontweight="bold")
    ax.text(MX + 1.2, my + 4.4, desc, fontsize=5.8, ha="left", va="center", linespacing=1.25)
    for k, t in enumerate(reversed(ts)):
        chip(MX + MW - 3.2 - k * 4.7, top - 3.2, t, C_ST[t])
    arrow(X_IN + W_INBOX + 0.6, 50, MX - 0.6, my + MH / 2)
    arrow(MX + MW + 0.6, my + MH / 2, 49.9, 64)

# ---- column 3: blend, with the ST3 column adjustments as a dashed step below ----
BX, BW = 50.5, 14.0
header(BX + BW / 2, "BLEND")
box(BX, 50, BW, 28, "white", lw=0.9)
ax.text(BX + BW / 2, 64, "equal-weight\nmean of member\nprobability matrices\n(no fitted weights)",
        fontsize=6.2, ha="center", va="center", linespacing=1.35)
box(BX, 9, BW, 31, "white", ls=(0, (2, 1.4)))
ax.text(BX + BW / 2, 24.5,
        "ST3 column adjust:\nexhortation adds\ns3s, m3q32A;\nage-restricted from\nm3q32A/B/C, m1m",
        fontsize=5.6, ha="center", va="center", linespacing=1.3, color="#333333")
ax.plot([BX + BW / 2, BX + BW / 2], [40.8, 49.2], color=C_EDGE, lw=0.7, ls=(0, (2, 1.4)))

# ---- column 4: decision layer, one row per sub-task ----
DX, DW = 67.5, 22.5
header(DX + DW / 2, "DECISION LAYER")
box(DX, 4, DW, 88, "white", lw=1.1)
drows = [
    ("ST1", "prior-corrected argmax\n(tau 0.4); class 'other'\nsuppressed"),
    ("ST2", "per-class thresholds;\nstability and low-support\nguards; argmax if empty"),
    ("ST3", "disclosure hard rules,\nthen thresholds, then\nconstraint cascade"),
]
RH, RGAP = 26.0, 2.0
for i, (t, txt) in enumerate(drows):
    top = 90.5 - i * (RH + RGAP)
    ry = top - RH
    box(DX + 1.2, ry, DW - 2.4, RH, "#fbfbfb", lw=0.5)
    chip(DX + 4.2, top - 4.2, t, C_ST[t], fs=6.0)
    ax.text(DX + 2.4, ry + 10.0, txt, fontsize=5.8, ha="left", va="center", linespacing=1.3)
arrow(BX + BW + 0.6, 64, DX - 0.6, 64)

# ---- column 5: output ----
OX, OW = 93.0, 6.5
header(OX + OW / 2, "OUTPUT")
box(OX, 28, OW, 44, "white")
for yy, t, txt in ((62, "ST1", "label"), (49, "ST2", "set"), (36, "ST3", "set")):
    chip(OX + OW / 2, yy + 3.4, t, C_ST[t], fs=5.4)
    ax.text(OX + OW / 2, yy - 2.6, txt, fontsize=5.8, ha="center", va="center")
arrow(DX + DW + 0.6, 50, OX - 0.6, 50)

fig.subplots_adjust(left=0.002, right=0.998, top=0.995, bottom=0.005)
fig.savefig(os.path.join(OUT, "pipeline.pdf"))
if len(sys.argv) > 1:
    fig.savefig(os.path.join(sys.argv[1], "pipeline_preview.png"), dpi=220)
print("wrote fig/pipeline.pdf")
