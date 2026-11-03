"""One card per (case, form) for followup D (cumulative component patching), with followup A
(joint patching) on top as context. Left = denoise (run N, insert P), right = noise (run P, insert N).

Row 1 (context, all 32 layers): residual patched at ONE layer at one segment (diff, result_tail,
  question, template, final; first run, results/posneg) and at diff + result_tail at once
  (followup A, layers 0-16). Shaded = layers where neither diff alone nor tail alone reaches
  0.85 in some direction (from followup A). diff / tail differ from followup A's by <= 0.07.
Rows 2-3 (D): every write of one component class into result_tail (row 2) or diff (row 3),
  over layers 0..L, replaced by the other run's. x = L. `all` = every write (reference).

Run (repo root): python plot_cumulative_cards.py -> results/followup/Qwen3.5-9B/cards_D/<case>__<form>.png
"""
import json, sys
from pathlib import Path
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from plot_posneg import INK, INK2
SURF, GRAY, GRID, SHADE = "#fcfcfb", "#52514e", "#d9d8d3", "#f0efec"
# categorical slots 1-5 = positions (row 1), 6-8 = component classes (rows 2-3)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
ROOT = Path("results/followup/Qwen3.5-9B")
J = pd.DataFrame(map(json.loads, open(ROOT / "joint.jsonl")))
R = pd.DataFrame(map(json.loads, open("results/posneg/Qwen3.5-9B/patching.jsonl")))
D = pd.DataFrame(map(json.loads, open(ROOT / "cumulative.jsonl")))
OUTDIR = ROOT / "cards_D"; OUTDIR.mkdir(exist_ok=True)
DIRS = [("denoise", "denoise: run N, insert P"), ("noise", "noise: run P, insert N")]
MAXL = 12
SEG_LINES = [("diff", BLUE, "diff"), ("result_tail", ORANGE, "result_tail"), ("question", AQUA, "question"),
             ("template", YELLOW, "template"), ("final", MAGENTA, "final")]
D_LINES = [("all", GRAY, "all writes (reference)"), ("deltanet", GREEN, "DeltaNet only"),
           ("attn", VIOLET, "full attention only"), ("mlp", RED, "MLP only")]


def clean(ax, maxl=MAXL):
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0, labelsize=7, colors=INK2)
    ax.axhline(1, color=GRID, lw=1, zorder=0); ax.axhline(0, color=GRID, lw=1, zorder=0)
    ax.set_ylim(-0.35, 1.3); ax.set_xlim(-0.3, maxl + 0.3)
    ax.set_xticks(range(0, maxl + 1, 1 if maxl <= MAXL else 2))


def conflict_layers(j):
    out = []
    for l in range(MAXL + 1):
        for d, _ in DIRS:
            x = j[(j.direction == d) & (j.layer == l)].set_index("positions").recovery
            if x["diff"] < 0.85 and x["result_tail"] < 0.85:
                out.append(l); break
    return out


def card(case, form):
    j = J[(J.case_id == case) & (J.form == form) & (J.layer <= MAXL)]
    jb = J[(J.case_id == case) & (J.form == form) & (J.positions == "both")]
    r = R[(R.case_id == case) & (R.form == form)]
    d = D[(D.case_id == case) & (D.form == form)]
    shade = conflict_layers(j)
    fig, axes = plt.subplots(3, 2, figsize=(11, 11.5), facecolor=SURF, sharey=True,
                             gridspec_kw=dict(hspace=0.5, wspace=0.06, height_ratios=[1.3, 1, 1]))
    for k, (dr, lab) in enumerate(DIRS):
        # row 1: A context
        ax = axes[0, k]
        for l in shade: ax.axvspan(l - 0.5, l + 0.5, color=SHADE, zorder=0)
        y = jb[jb.direction == dr].sort_values("layer")
        ax.plot(y.layer, y.recovery, color=GRAY, lw=2, ls="--", marker="o", ms=3, label="diff + result_tail at once")
        for seg, col, name in SEG_LINES:
            y = r[(r.direction == dr) & (r.segment == seg)].sort_values("layer")
            ax.plot(y.layer, y.recovery, color=col, lw=2, marker="o", ms=3, label=name)
            pk = y.loc[y.recovery.idxmax()]
            ax.annotate(name, (pk.layer, pk.recovery), xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=7, color=INK2)
        clean(ax, 31)
        ax.set_title(f"A (context). one layer, residual  |  {lab}", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("patched layer   (shaded = neither diff alone nor tail alone reaches 0.85)", fontsize=8, color=INK2)
        # rows 2-3: D
        for row, pos in [(1, "result_tail"), (2, "diff")]:
            ax = axes[row, k]
            for l in shade: ax.axvspan(l - 0.5, l + 0.5, color=SHADE, zorder=0)
            for comp, col, name in D_LINES:
                y = d[(d.direction == dr) & (d.positions == pos) & (d.component == comp)].sort_values("up_to_layer")
                ax.plot(y.up_to_layer, y.recovery, color=col, lw=3.5 if comp == "all" else 2,
                        alpha=0.45 if comp == "all" else 1, marker="o", ms=4, label=name)
            clean(ax)
            ax.set_title(f"D. writes into {pos}, layers 0..L  |  {lab}", fontsize=9, color=INK, loc="left")
            ax.set_xlabel("L  (measured at 3, 5, 6, 7, 8, 9, 10, 12; full attention = layers 3, 7, 11)",
                          fontsize=8, color=INK2)
    for r in range(3): axes[r, 0].set_ylabel("recovery", fontsize=8, color=INK2)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=6, fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.935),
               labelcolor=INK2, title="row 1 = which segment is patched", title_fontsize=7.5)
    axes[1, 0].legend(fontsize=7, frameon=True, framealpha=0.9, edgecolor="none", facecolor=SURF,
                      loc="center left", labelcolor=INK2)
    m = j.iloc[0]
    fig.suptitle(f"{case}  |  {form}   m_P {m.m_P:.2f}, m_N {m.m_N:.2f}", fontsize=11, color=INK, y=0.965)
    fig.savefig(OUTDIR / f"{case}__{form}.png", dpi=140, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


for case in sorted(D.case_id.unique()):
    for form in ["numeric", "lexical", "negation"]:
        card(case, form)
print(f"wrote {OUTDIR}")
