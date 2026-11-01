"""Summarize a patch_posneg.py run: behavior table + layer x segment recovery heatmaps.

Behavior (selected cases only), per test x form, on the metric
m = logit(target) - logsumexp(other three letters):
    up   = m_P - m_B   how far the POSITIVE result pushes the target up from baseline
    down = m_B - m_N   how far the NEGATIVE result pushes it down
"Ignoring negatives" shows up as down << up.

Heatmaps: mean recovery over selected (case, form) pairs whose P-N gap >= min_gap (empty
cells / n=0 when a form never separates P from N), one row per form, columns = denoise (run N, insert P) | noise (run P, insert N). Layers marked with
"*" are full-attention layers; the rest are linear-attention (Gated DeltaNet).

Run (repo root):
  python plot_posneg.py \
      --run results/posneg/Qwen3.5-9B
Output (in --run): behavior_summary.md, heatmap_recovery.png, heatmap_recovery_<test>.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

FORMS = ["numeric", "lexical", "negation"]
DIRS = [("denoise", "denoise: run N, insert P"), ("noise", "noise: run P, insert N")]
# diverging blue <-> gray <-> red (dataviz reference palette)
CMAP = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
INK, INK2 = "#0b0b0b", "#52514e"


def behavior_table(beh, sel_ids):
    rows = []
    for r in beh:
        if r["case_id"] not in sel_ids:
            continue
        for f in FORMS:
            rows.append({"test": r["test_id"], "form": f, "m_B": r["B"]["m"],
                         "m_P": r[f"{f}_P"]["m"], "m_N": r[f"{f}_N"]["m"],
                         "letter_mass": np.mean([r["B"]["letter_mass"], r[f"{f}_P"]["letter_mass"],
                                                 r[f"{f}_N"]["letter_mass"]])})
    df = pd.DataFrame(rows)
    df["up"] = df.m_P - df.m_B
    df["down"] = df.m_B - df.m_N
    g = df.groupby(["test", "form"]).agg(n=("m_B", "size"), m_B=("m_B", "mean"), m_P=("m_P", "mean"),
                                         m_N=("m_N", "mean"), up=("up", "mean"), down=("down", "mean"),
                                         letter_mass=("letter_mass", "mean")).reset_index()
    allf = df.groupby("form").agg(n=("m_B", "size"), m_B=("m_B", "mean"), m_P=("m_P", "mean"),
                                  m_N=("m_N", "mean"), up=("up", "mean"), down=("down", "mean"),
                                  letter_mass=("letter_mass", "mean")).reset_index()
    allf.insert(0, "test", "ALL")
    return pd.concat([g, allf], ignore_index=True)


def heatmap(df, segments, n_layers, layer_types, title, path):
    fig, axes = plt.subplots(len(FORMS), 2, figsize=(9, 4.2 * len(FORMS)), sharey=True)
    ylabels = [f"{l}{'*' if layer_types and layer_types[l] == 'full_attention' else ''}"
               for l in range(n_layers)]
    for i, form in enumerate(FORMS):
        for j, (d, dlabel) in enumerate(DIRS):
            ax = axes[i, j]
            sub = df[(df.form == form) & (df.direction == d)]
            piv = sub.groupby(["layer", "segment"]).recovery.mean().unstack("segment")
            piv = piv.reindex(index=range(n_layers), columns=segments)
            im = ax.imshow(piv.values, origin="lower", aspect="auto", cmap=CMAP, vmin=-1, vmax=1)
            ax.set_xticks(range(len(segments)), segments, rotation=30, ha="right", fontsize=8, color=INK2)
            ax.set_yticks(range(n_layers), ylabels, fontsize=6, color=INK2)
            ax.set_title(f"{form} (n={sub.case_id.nunique()}) | {dlabel}", fontsize=9, color=INK)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.tick_params(length=0)
        axes[i, 0].set_ylabel("layer (* = full attention)", fontsize=8, color=INK2)
    cb = fig.colorbar(im, ax=axes, shrink=0.5, pad=0.02)
    cb.set_label("recovery = fraction of the P-N gap moved", fontsize=8, color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle(title, fontsize=10, color=INK)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    cfg = json.loads((run / "run_config.json").read_text())
    beh = [json.loads(l) for l in open(run / "behavior.jsonl")]
    sel = json.loads((run / "selected.json").read_text())
    pat = pd.DataFrame([json.loads(l) for l in open(run / "patching.jsonl")])

    sel_ids = {s["case_id"] for s in sel}
    n_elig = sum(s["eligible"] for s in sel)
    # heatmaps only from (case, form) pairs with a usable P-N gap
    gap_n = pat.drop_duplicates(["case_id", "form"]).groupby("form").gap_ok.agg(["sum", "size"])
    pat = pat[pat.case_id.isin(sel_ids) & pat.gap_ok]
    segs = [s for s in cfg["segments"] if s in set(pat.segment)]

    bt = behavior_table(beh, sel_ids)
    md = [f"# Positive vs negative evidence: {Path(cfg['model']).name}", "",
          f"Selected cases: {len(sel)} ({n_elig} with baseline P(target) in {cfg['pB_range']}).", "",
          "m = logit(target letter) - logsumexp(other letters). up = m_P - m_B; down = m_B - m_N. "
          "letter_mass = probability the model puts on A-D at all (should be near 1).", "",
          bt.to_markdown(index=False, floatfmt=".2f"), "",
          f"Pairs used in patching heatmaps (P-N gap >= {cfg['min_gap']}): "
          + ", ".join(f"{f} {int(gap_n.loc[f, 'sum'])}/{int(gap_n.loc[f, 'size'])}"
                      for f in FORMS if f in gap_n.index), ""]
    # patching: strongest site per form x direction
    md += ["## Patching: top layer x segment per form and direction (mean recovery)", ""]
    top = (pat.groupby(["form", "direction", "segment", "layer"]).recovery.mean().reset_index()
           .sort_values("recovery", ascending=False).groupby(["form", "direction"]).head(3))
    md += [top.to_markdown(index=False, floatfmt=".2f"), ""]
    (run / "behavior_summary.md").write_text("\n".join(md))

    lt = cfg.get("layer_types")
    heatmap(pat, segs, cfg["n_layers"], lt,
            f"{Path(cfg['model']).name}: residual patching, mean over cases with P-N gap >= {cfg['min_gap']} (n per panel)",
            run / "heatmap_recovery.png")
    for t in sorted(pat.test_id.unique()):
        heatmap(pat[pat.test_id == t], segs, cfg["n_layers"], lt,
                f"{Path(cfg['model']).name}: {t}", run / f"heatmap_recovery_{t}.png")
    print(f"wrote {run}/behavior_summary.md and heatmaps")


if __name__ == "__main__":
    main()
