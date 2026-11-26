"""Two figures for the DDW 2027 abstract. Aggregate numbers only, copied from paper/main.tex
(Table 1 and Figure 2); no MIMIC text. Run: python paper/ddw_figures.py"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullLocator

OUT = Path(__file__).parent
BLUE, ORANGE, TEAL, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#ffffff"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.dpi": 300,
})

# ---------- Figure 1: shift relative to LR-predicted shift (Table 1, 94 held-out histories) ----------
tests = ["CT, diverticulitis", "CT, kidney stone", "CT, appendicitis",
         "Ultrasound, cholecystitis", "Ultrasound, appendicitis",
         "Serum lipase, pancreatitis", "Serum amylase, pancreatitis"]
up = [1.01, 1.81, 1.66, 3.26, 2.18, 1.22, 1.27]
down = [0.33, 0.58, 0.68, 1.49, 0.97, 1.81, 1.58]

fig, ax = plt.subplots(figsize=(7.2, 4.3))
gap = 0.6  # extra space between imaging and lab groups
y = np.array([6, 5, 4, 3, 2, 1 - gap, 0 - gap]) + gap
h = 0.36
ax.barh(y + h / 2 + 0.01, up, height=h, color=ORANGE, label="Positive report", zorder=3)
ax.barh(y - h / 2 - 0.01, down, height=h, color=BLUE, label="Negative report", zorder=3)
for yi, d in zip(y, down):
    ax.text(d + 0.05, yi - h / 2 - 0.01, f"{d:.2f}", va="center", fontsize=8.5, color=INK)
ax.axvline(1, color=INK, lw=1.2, ls=(0, (4, 3)), zorder=4)
ax.text(1.03, y.max() + 0.5, "LR-predicted shift", fontsize=8.5, color=INK, va="bottom")
ax.set_yticks(y, tests)
ax.set_xlim(0, 3.5)
ax.set_ylim(-0.6, y.max() + 0.9)
ax.set_xlabel("Shift in belief ÷ shift predicted by literature likelihood ratio")
ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
ax.tick_params(axis="y", length=0)
ax.axhline((y[4] + y[5]) / 2, color=GRID, lw=1)
ax.text(3.48, y[0] + 0.5, "Imaging", ha="right", fontsize=9, color=INK2, style="italic")
ax.text(3.48, y[5] + 0.5, "Laboratory", ha="right", fontsize=9, color=INK2, style="italic")
ax.legend(loc="lower left", frameon=False, fontsize=9, ncol=2, bbox_to_anchor=(0.0, 1.0), borderaxespad=0.2)
ax.set_title("Negative imaging moves the model less than its positives; lab negatives do not",
             fontsize=10.5, loc="left", color=INK, pad=26)
fig.tight_layout()
fig.savefig(OUT / "ddw_fig1_shifts.png")

# ---------- Figure 2: imbalance before/after repair on held-out set (paper Figure 2) ----------
# per-test mean |asym| -> fold imbalance exp(mean |asym|); 1 = balanced as the LRs predict
tests2 = ["CT, diverticulitis", "CT, kidney stone*", "CT, appendicitis",
          "Ultrasound, cholecystitis*", "Ultrasound, appendicitis*",
          "Serum lipase", "Serum amylase*"]
orig = np.exp([1.41, 1.14, 0.91, 0.81, 0.79, 0.40, 0.22])
rep = np.exp([0.32, 0.42, 0.25, 0.18, 0.28, 0.40, 0.28])

fig, ax = plt.subplots(figsize=(7.2, 4.3))
y = np.array([6, 5, 4, 3, 2, 1 - gap, 0 - gap]) + gap
for yi, o, r in zip(y, orig, rep):
    ax.plot([min(o, r), max(o, r)], [yi, yi], color=GRID, lw=3, zorder=2, solid_capstyle="round")
ax.scatter(orig, y, s=70, color=GRAY, edgecolor=SURF, linewidth=2, zorder=3, label="Original model")
ax.scatter(rep, y, s=70, color=TEAL, edgecolor=SURF, linewidth=2, zorder=4, label="After repair")
for yi, o, r in zip(y, orig, rep):
    if abs(o - r) < 0.01:
        ax.text(r * 1.04, yi + 0.3, f"{r:.2f}× (unchanged)", fontsize=8.5, color=INK, va="center")
        continue
    ax.text(o * 1.04 if o >= r else o / 1.04, yi + 0.3, f"{o:.2f}×", fontsize=8.5, color=INK2,
            ha="left" if o >= r else "right", va="center")
    ax.text(r / 1.04 if o >= r else r * 1.04, yi + 0.3, f"{r:.2f}×", fontsize=8.5, color=INK,
            ha="right" if o >= r else "left", va="center")
ax.set_xscale("log")
ticks = [1, 1.5, 2, 3, 4]
ax.xaxis.set_major_locator(FixedLocator(ticks))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticklabels([f"{t:g}×" for t in ticks])
ax.set_xlim(0.95, 5)
ax.set_yticks(y, tests2)
ax.set_ylim(-0.6, y.max() + 0.7)
ax.tick_params(axis="y", length=0)
ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
ax.axhline((y[4] + y[5]) / 2, color=GRID, lw=1)
ax.set_xlabel("Positive/negative imbalance vs. likelihood-ratio prediction (1× = balanced)")
ax.legend(loc="lower right", frameon=False, fontsize=9)
ax.set_title("Held-out patients: overall imbalance 2.5× → 1.35× (random edits: 2.4–2.5×)",
             fontsize=10.5, loc="left", color=INK, pad=10)
fig.text(0.01, 0.01, "* test never seen during the repair search. Wrong-direction negative updates: 79 → 2.",
         fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT / "ddw_fig2_repair.png")
print("saved")
