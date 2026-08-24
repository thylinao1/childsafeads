"""Figures for the system description paper. Writes PDFs into paper/latex/fig/,
created on demand; the paper source tree is not part of this repository."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "eval")); sys.path.insert(0, os.path.join(ROOT, "decide"))
sys.path.insert(0, os.path.join(ROOT, "clinic"))
from local_scorer import ST1  # noqa: E402
import decision_layer as DL   # noqa: E402
import level_ablation as LA   # noqa: E402

OUT = os.path.join(ROOT, "paper", "latex", "fig")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9.5, "font.family": "serif",
                     "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
                     "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})

# ---- Figure 1: the bimodal bootstrap distribution of the reported metric ----
params = json.load(open(os.path.join(ROOT, "decide", "params.json")))
rows = DL.emit_preds(params["members"], "oof", params, params.get("weights"))
preds = {r["instanceID"]: r for r in rows}
gold = {json.loads(l)["instanceID"]: json.loads(l)
        for l in open(os.path.join(ROOT, "data", "train.jsonl")) if l.strip()}
ids = sorted(gold)
blocks = LA._indicator_blocks(gold, preds, ids)
channels, agg = LA._channel_agg(blocks, ids, gold)
nC = len(channels)
j = ST1.index("other")
TP, FP, FN = agg["st1"]
rare_per_ch = TP[:, j] + FN[:, j]

rng = np.random.default_rng(0); reps = 8000
draws = rng.integers(0, nC, size=(reps, nC))
counts = np.zeros((reps, nC))
for r in range(reps):
    counts[r] = np.bincount(draws[r], minlength=nC)
cols = {t: LA._macro_from_counts(*[counts @ M for M in agg[t]]) for t in ("st1", "st2", "st3")}
metric = sum(cols.values()) / 3
has = (counts @ rare_per_ch) > 0

fig, ax = plt.subplots(figsize=(2.78, 2.05))
bins = np.linspace(metric.min(), metric.max(), 70)
ax.hist(metric[has], bins=bins, color="#3b5f9e", alpha=0.95,
        label=f"gold `other` drawn ({100*has.mean():.0f}%)")
ax.hist(metric[~has], bins=bins, color="#d9a441", alpha=0.95, hatch="///", edgecolor="#7a5a10", linewidth=0.0,
        label=f"not drawn ({100*(~has).mean():.0f}%)")
for m, c, st in ((metric[has].mean(), "#3b5f9e", "--"), (metric[~has].mean(), "#7a5a10", ":")):
    ax.axvline(m, color=c, lw=1.1, ls=st)
ax.annotate("", xy=(metric[~has].mean(), 325), xytext=(metric[has].mean(), 325),
            arrowprops=dict(arrowstyle="<->", lw=0.7, color="black"))
ax.text((metric[has].mean() + metric[~has].mean()) / 2, 350,
        f"{metric[~has].mean()-metric[has].mean():+.3f}", ha="center", fontsize=8)
ax.set_xlabel("mean macro-F1, channel resample of the out-of-fold split")
ax.set_ylabel("replicates")
ax.set_ylim(0, 520)
ax.legend(frameon=False, fontsize=8.5, loc="upper center", handlelength=1.1, borderaxespad=0.2)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.25)
fig.savefig(os.path.join(OUT, "divisor.pdf"), bbox_inches="tight", pad_inches=0.02)
print("wrote fig/divisor.pdf  |  regimes:",
      round(float(metric[has].mean()), 4), round(float(metric[~has].mean()), 4),
      "gap", round(float(metric[~has].mean() - metric[has].mean()), 4))

# ---- Figure 2: data access levels ----
la = json.load(open(os.path.join(ROOT, "clinic", "level_ablation.json")))
sc = la["scores"]
levels = ["1", "2", "3", "4"]
names = ["L1\ntranscript", "L2\n+video", "L3\n+channel", "L4\n+page"]
fig, ax = plt.subplots(figsize=(2.78, 1.9))
w = 0.26
x = np.arange(4)
for k, (task, col) in enumerate((("st1", "#4c72b0"), ("st2", "#55a868"), ("st3", "#c44e52"))):
    ax.bar(x + (k - 1) * w, [sc[l]["oof"][task] for l in levels], w, color=col,
           label=task.upper().replace("ST", "ST"))
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel("macro-F1 (out of fold)")
ax.set_ylim(0, 0.85)
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left", handlelength=1.2)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.25)
fig.savefig(os.path.join(OUT, "levels.pdf"), bbox_inches="tight", pad_inches=0.02)
print("wrote fig/levels.pdf")
