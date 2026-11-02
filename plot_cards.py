"""One card per (case, form): A. where (residual patching, layer x segment), B. diff vs
result_tail alone (residual patching, layers 0-16), C. single component, single layer
(mechanism run). Left = denoise (run N, insert P), right = noise (run P, insert N).

Run (repo root): python plot_cards.py   -> results/mech/Qwen3.5-9B/cards/<case>__<form>.png
"""
import json, sys
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from plot_posneg import CMAP, INK, INK2
SURF, BLUE, ORANGE, GRAY = "#fcfcfb", "#2a78d6", "#eb6834", "#52514e"
R_ALL = pd.DataFrame(map(json.loads, open("results/posneg/Qwen3.5-9B/patching.jsonl")))
C_ALL = pd.DataFrame(map(json.loads, open("results/mech/Qwen3.5-9B/components.jsonl")))
OUTDIR = Path("results/mech/Qwen3.5-9B/cards"); OUTDIR.mkdir(exist_ok=True)
segs = ["diff", "result_tail", "question", "template", "final"]
lt = json.load(open("results/posneg/Qwen3.5-9B/run_config.json"))["layer_types"]
star = lambda l: f"{l}*" if lt[l] == "full_attention" else str(l)
DIRS = [("denoise", "denoise: run N, insert P"), ("noise", "noise: run P, insert N")]

def card(CASE, FORM):
    r = R_ALL[(R_ALL.case_id == CASE) & (R_ALL.form == FORM)]
    c = C_ALL[(C_ALL.case_id == CASE) & (C_ALL.form == FORM)]
    fig = plt.figure(figsize=(11, 13), facecolor=SURF)
    gs = fig.add_gridspec(3, 2, height_ratios=[2.2, 1.2, 1.0], hspace=0.45, wspace=0.08)
    def clean(ax):
        for s in ax.spines.values(): s.set_visible(False)
        ax.tick_params(length=0, labelsize=7, colors=INK2)

    # A. where: residual patching, layer x segment
    for j, (d, lab) in enumerate(DIRS):
        ax = fig.add_subplot(gs[0, j])
        piv = r[r.direction == d].pivot_table(index="layer", columns="segment", values="recovery")[segs]
        im = ax.imshow(piv.values, origin="lower", aspect="auto", cmap=CMAP, vmin=-1, vmax=1)
        ax.set_xticks(range(5), segs, rotation=25, ha="right"); ax.set_yticks(range(32), [star(l) for l in range(32)])
        ax.set_title(f"A. where (whole residual)  |  {lab}", fontsize=9, color=INK, loc="left"); clean(ax)
        if j == 0: ax.set_ylabel("layer (* = full attention)", fontsize=8, color=INK2)
    cb = fig.colorbar(im, ax=[fig.axes[0], fig.axes[1]], shrink=0.6, pad=0.02); cb.outline.set_visible(False)
    cb.set_label("recovery", fontsize=8, color=INK2); cb.ax.tick_params(labelsize=7, length=0)

    # B. diff alone vs result_tail alone: diff alone, tail alone, sum
    for j, (d, lab) in enumerate(DIRS):
        ax = fig.add_subplot(gs[1, j])
        x = r[r.direction == d].pivot_table(index="layer", columns="segment", values="recovery").loc[0:16]
        ax.axhline(1, color="#d9d8d3", lw=1); ax.axhline(0, color="#d9d8d3", lw=1)
        ax.axvspan(5.5, 9.5, color="#f0efec", zorder=0)
        for y, col, ls, name in [(x["diff"], BLUE, "-", "diff alone"), (x["result_tail"], ORANGE, "-", "result_tail alone")]:
            ax.plot(y.index, y.values, color=col, lw=2, ls=ls, marker="o", ms=3, label=name)
        ax.legend(fontsize=7, frameon=False, loc="upper right", labelcolor=INK2)
        ax.set_xlim(0, 16.5); ax.set_ylim(-0.2, 1.3); ax.set_xticks(range(0, 17, 2))
        ax.set_title(f"B. diff alone vs result_tail alone  |  {lab}", fontsize=9, color=INK, loc="left"); clean(ax)
        ax.set_xlabel("layer (shaded: 6-9)", fontsize=8, color=INK2)
        if j == 0: ax.set_ylabel("recovery", fontsize=8, color=INK2)

    # C. which part: single component, single layer
    rows = [("diff", "mixer"), ("diff", "mlp"), ("result_tail", "mixer"), ("result_tail", "mlp")]
    for j, (d, lab) in enumerate(DIRS):
        ax = fig.add_subplot(gs[2, j])
        p = c[c.direction == d].pivot_table(index=["segment", "component"], columns="layer", values="recovery").reindex(rows)
        ax.imshow(p.values, aspect="auto", cmap=CMAP, vmin=-1, vmax=1)
        for i in range(p.shape[0]):
            for k in range(p.shape[1]):
                v = p.values[i, k]; ax.text(k, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color=INK)
        ax.set_yticks(range(4), [f"{s} / {cpt}" for s, cpt in rows] if j == 0 else [""] * 4); ax.set_xticks(range(p.shape[1]), [star(l) for l in p.columns])
        ax.set_title(f"C. which part (one component, one layer)  |  {lab}", fontsize=9, color=INK, loc="left"); clean(ax)
        ax.set_xlabel("layer  (mixer = DeltaNet, or attention on * layers)", fontsize=8, color=INK2)
    fig.suptitle(f"{CASE}  |  {FORM}   m_P {r.m_P.iloc[0]:.2f}, m_N {r.m_N.iloc[0]:.2f}", fontsize=11, color=INK, y=0.93)
    fig.savefig(OUTDIR / f"{CASE}__{FORM}.png", dpi=140, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


for CASE in sorted(C_ALL.case_id.unique()):
    for FORM in ["numeric", "lexical", "negation"]:
        card(CASE, FORM)
print(f"wrote {OUTDIR}")
