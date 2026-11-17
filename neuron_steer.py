"""Steer the candidate neurons on our own P/N questions: does pushing them change the answer?

For each chosen site (segment, layer, neuron) the neuron's value at the segment's positions is set to
    host + alpha * (other - host)
with host / other = the clean values of this case's N and P runs (same token positions, P and N are
token-aligned). Both are taken from one clean float32 forward, as in neuron_posneg.py stage 3.
    host N, alpha > 0   push the N question toward its P version (does the answer move to the target?)
    host P, alpha > 0   push the P question toward its N version (does the answer leave the target?)
    alpha = 1           = patching (neuron_posneg.py); alpha 2, 5, 10 go past the natural range
    alpha = -1          push the other way (control)
    alpha = 0           no change (sanity check: must equal the clean run)
Each neuron follows its own measured P-N difference, so "against P" neurons are pushed the way they
naturally move too; the group effect is their net effect.

Groups (sites from top_neurons.csv, scope "all", ranking "largest"):
    candidates   the ids in candidates_largest.txt, at the segment they were found in
    top10        the 10 highest-scoring sites (the group patched in neuron_posneg.py, recovery ~0.2)
    random       same segment and layer as each group site, random neuron index (--random-draws per group)
    single       each candidate site alone
Cases: half B only (never used for ranking; held-out tests are all in B), float32, pairs with
gap = m_P - m_N >= --min-gap. m = logit(target letter) - logsumexp(other 3 letters).
    shift   (m_steered - m_host) / gap, sign so + = toward the other run: 1 = moved as far as the
            other version is
    answer  argmax over the 4 letters; flip = N host: not target -> target; P host: target -> not target

Output: results/neuron/<model_tag>/steer/
    steer_rows.jsonl   one row per pair x host x group x alpha
    steer_summary.csv  mean / median shift, answer == target rate, flip rate; all, non-held-out, held-out
    run_config.json

Run (repo root):
    python -u neuron_steer.py --model Qwen/Qwen3-14B                        # 1 GPU with 80 GB, float32
    python neuron_steer.py --model Qwen/Qwen3-0.6B --fake-sites --pairs 4 --device cpu   # local smoke test
"""
import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from patch_posneg import FORMS, LETTERS, SEGMENTS, encode, metric, segments

HOSTS = ["N", "P"]


def half_b_cases(pool_path, patients_per_test, holdout_tests):
    """Same split as neuron_posneg.py: per test, patients alternate A/B; held-out tests all B."""
    per_test = {}
    for line in open(pool_path):
        c = json.loads(line)
        per_test.setdefault(c["test_id"], []).append(c)
    out = []
    for t, cases in per_test.items():
        patients = list(dict.fromkeys(c["hadm_id"] for c in cases))[:patients_per_test]
        halves = {h: "B" if t in holdout_tests or i % 2 else "A" for i, h in enumerate(patients)}
        out += [c for c in cases if halves.get(c["hadm_id"]) == "B"]
    return out


def load_groups(csv_path, cand_path, n_top):
    """Sites (segment, layer, neuron) for the candidates and the top-n_top group."""
    rows = [r for r in csv.DictReader(open(csv_path)) if r["scope"] == "all" and r["ranking"] == "largest"]
    rows.sort(key=lambda r: -float(r["score_A"]))
    site = lambda r: (r["segment"], int(r["layer"]), int(r["index"]))
    wanted = Path(cand_path).read_text().split()
    cand = []
    for nid in wanted:              # the segment(s) where this id passed the candidate filter
        hits = [site(r) for r in rows if r["neuron"] == nid and r["above_null_p95"] == "True"
                and float(r["mean_cell_B"]) > 0]
        if not hits:
            raise SystemExit(f"{nid} not found as a candidate row in {csv_path}")
        cand += [h for h in hits if h not in cand]
    top = list(dict.fromkeys(site(r) for r in rows))[:n_top]
    return {"candidates": cand, f"top{n_top}": top}


@torch.no_grad()
def run_rows(hf, blocks, ids, host, other, rows, seg, batch, letter_ids, ti, device):
    """Each row: dict(sites, alpha). host/other: {layer: [T, N]} clean values. Returns [(m, letter)]."""
    out = []
    for start in range(0, len(rows), batch):
        chunk = rows[start: start + batch]
        idx = defaultdict(lambda: ([], [], [], []))
        for r, row in enumerate(chunk):
            for s, l, n in row["sites"]:
                for p in seg.get(s, []):
                    a, b, c, d = idx[l]
                    a.append(r), b.append(p), c.append(n), d.append(row["alpha"])
        ix = {l: [torch.tensor(v, device=device) for v in vs[:3]] + [torch.tensor(vs[3], device=device, dtype=torch.float32)]
              for l, vs in idx.items()}

        def make(l):
            r_, p_, n_, a_ = ix[l]
            def hook(module, args):
                x = args[0].clone()
                h, o = host[l][p_, n_], other[l][p_, n_]
                x[r_, p_, n_] = (h + a_ * (o - h)).to(x.dtype)
                return (x,)
            return hook

        handles = [blocks[l].mlp.down_proj.register_forward_pre_hook(make(l)) for l in ix]
        try:
            logits = hf(input_ids=torch.tensor([ids] * len(chunk), device=device), logits_to_keep=1,
                        use_cache=False).logits[:, -1, :]
        finally:
            for h in handles:
                h.remove()
        out += list(zip(metric(logits, letter_ids, ti).tolist(), logits[:, letter_ids].argmax(-1).tolist()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--pool", default="data/posneg_pool_v2.jsonl")
    ap.add_argument("--screen-dir", default=None, help="neuron_posneg.py output (default results/neuron/<tag>)")
    ap.add_argument("--candidates", default="candidates_largest.txt")
    ap.add_argument("--n-top", type=int, default=10)
    ap.add_argument("--alphas", type=float, nargs="+", default=[-1, 1, 2, 5, 10])
    ap.add_argument("--random-draws", type=int, default=3)
    ap.add_argument("--no-singles", action="store_true")
    ap.add_argument("--pairs", type=int, default=120, help="random half-B pairs (0 = all)")
    ap.add_argument("--patients-per-test", type=int, default=10)
    ap.add_argument("--holdout-tests", nargs="*", default=["us_gallbladder", "hida"])
    ap.add_argument("--min-gap", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fake-sites", action="store_true", help="smoke test on another model: random valid sites")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    tag = args.model.split("/")[-1]
    screen = Path(args.screen_dir or f"results/neuron/{tag}")
    out = Path("results/neuron") / tag / "steer"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device).eval()
    for p in hf.parameters():
        p.requires_grad_(False)
    blocks = hf.model.layers
    n_layers, n_neurons = len(blocks), blocks[0].mlp.down_proj.in_features
    letter_ids = tok.convert_tokens_to_ids(LETTERS)

    if args.fake_sites:
        segs = ["question", "template", "final"]
        fake = lambda k: [(segs[i % 3], int(rng.integers(n_layers)), int(rng.integers(n_neurons))) for i in range(k)]
        groups = {"candidates": fake(10), f"top{args.n_top}": fake(args.n_top)}
    else:
        groups = load_groups(screen / "top_neurons.csv", screen / args.candidates, args.n_top)
    for name, sites in groups.items():
        bad = [x for x in sites if x[1] >= n_layers or x[2] >= n_neurons]
        if bad:
            raise SystemExit(f"{name}: sites {bad} do not exist in {args.model}; ids are model-specific")
        print(f"{name}: " + ", ".join(f"{s}:L{l}_N{n}" for s, l, n in sites), flush=True)

    sets = [dict(group=g, kind="top", draw=0, sites=s) for g, s in groups.items()]
    for g, s in groups.items():
        for d in range(args.random_draws):
            sets.append(dict(group=g, kind="random", draw=d, sites=[(a, l, int(rng.integers(n_neurons))) for a, l, _ in s]))
    if not args.no_singles:
        sets += [dict(group=f"L{l}_N{n}", kind="single", draw=0, sites=[(s, l, n)]) for s, l, n in groups["candidates"]]
    plan = [dict(set=None, alpha=0.0, sites=[])]
    plan += [dict(set=i, alpha=a, sites=st["sites"]) for i, st in enumerate(sets) for a in args.alphas]
    print(f"{len(sets)} neuron sets x {len(args.alphas)} alphas + no-op = {len(plan)} rows per host", flush=True)

    cases = half_b_cases(args.pool, args.patients_per_test, args.holdout_tests)
    pairs = [(c, f) for c in cases for f in FORMS]
    if args.pairs and len(pairs) > args.pairs:
        pairs = [pairs[i] for i in sorted(rng.choice(len(pairs), args.pairs, replace=False))]

    records, noop_err, low_gap, skipped = [], 0.0, 0, 0
    t0 = time.time()
    store = {}
    with (out / "steer_rows.jsonl").open("w") as f:
        for done, (c, form) in enumerate(pairs, 1):
            ti = LETTERS.index(c["target_letter"])
            fm = c["forms"][form]
            _, p_ids, offs = encode(tok, fm["P"])
            _, n_ids, _ = encode(tok, fm["N"])
            if len(p_ids) != len(n_ids):
                skipped += 1
                continue
            seg = segments(tok, fm["P"], fm["P_result"], p_ids, n_ids, offs)

            def keep(l):
                def hook(module, args):
                    store[l] = args[0].detach()
                return hook
            handles = [b.mlp.down_proj.register_forward_pre_hook(keep(l)) for l, b in enumerate(blocks)]
            with torch.no_grad():
                clean = hf(input_ids=torch.tensor([p_ids, n_ids], device=device), logits_to_keep=1,
                           use_cache=False).logits[:, -1, :]
            for h in handles:
                h.remove()
            (m_p, m_n), (ans_p, ans_n) = metric(clean, letter_ids, ti).tolist(), clean[:, letter_ids].argmax(-1).tolist()
            gap = m_p - m_n
            if gap < args.min_gap:
                low_gap += 1
                continue
            vals = {"P": {l: v[0] for l, v in store.items()}, "N": {l: v[1] for l, v in store.items()}}
            store.clear()
            for host in HOSTS:
                other = "P" if host == "N" else "N"
                ids = n_ids if host == "N" else p_ids
                m_host, ans_host = (m_n, ans_n) if host == "N" else (m_p, ans_p)
                res = run_rows(hf, blocks, ids, vals[host], vals[other], plan, seg, args.batch, letter_ids, ti, device)
                noop_err = max(noop_err, abs(res[0][0] - m_host) / gap)
                for row, (m, ans) in zip(plan[1:], res[1:]):
                    st = sets[row["set"]]
                    shift = (m - m_host) / gap * (1 if host == "N" else -1)
                    rec = dict(case_id=c["case_id"], test_id=c["test_id"], form=form, held_out=c["test_id"] in args.holdout_tests,
                               host=host, group=st["group"], kind=st["kind"], draw=st["draw"], alpha=row["alpha"],
                               gap=gap, m_host=m_host, m=m, shift=shift,
                               base_target=ans_host == ti, steered_target=ans == ti)
                    records.append(rec)
                    f.write(json.dumps(rec) + "\n")
            print(f"  steering {done}/{len(pairs)} pairs ({time.time() - t0:.0f}s)", flush=True)

    agg = defaultdict(list)
    for r in records:
        for part in ("all", "held_out" if r["held_out"] else "not_held_out"):
            agg[(part, r["host"], r["kind"], r["group"], r["alpha"])].append(r)
    with (out / "steer_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cases", "host", "kind", "group", "alpha", "n", "mean_shift", "median_shift",
                    "base_target_rate", "steered_target_rate", "n_can_flip", "flip_rate"])
        for key, rs in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3], kv[0][4])):
            sh = np.array([r["shift"] for r in rs])
            can = [r for r in rs if r["base_target"] == (key[1] == "P")]
            flips = [r for r in can if r["steered_target"] != r["base_target"]]
            w.writerow([*key, len(rs), round(sh.mean(), 4), round(float(np.median(sh)), 4),
                        round(np.mean([r["base_target"] for r in rs]), 3), round(np.mean([r["steered_target"] for r in rs]), 3),
                        len(can), round(len(flips) / len(can), 3) if can else ""])
    (out / "run_config.json").write_text(json.dumps(dict(
        vars(args), device=device, groups={g: [list(s) for s in v] for g, v in groups.items()}, n_pairs_run=len(pairs) - low_gap - skipped,
        low_gap=low_gap, skipped=skipped, max_noop_error=noop_err), indent=2))

    print(f"no-op rows differ from clean by at most {noop_err:.5f} of the gap (should be ~0); "
          f"{low_gap} pairs dropped (gap < {args.min_gap}), {skipped} not token-aligned")
    for key, rs in sorted(agg.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][3], kv[0][4])):
        if key[0] == "all" and key[2] in ("top", "random"):
            can = [r for r in rs if r["base_target"] == (key[1] == "P")]
            fl = sum(r["steered_target"] != r["base_target"] for r in can)
            print(f"host {key[1]} {key[2]:7s} {key[3]:12s} alpha {key[4]:5.1f}  n={len(rs):4d}  "
                  f"mean shift {np.mean([r['shift'] for r in rs]):+.3f}  flips {fl}/{len(can)}")
    print(f"done ({time.time() - t0:.0f}s) -> {out}", flush=True)


if __name__ == "__main__":
    main()
