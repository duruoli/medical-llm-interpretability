"""Which MLP neurons carry the P-vs-N difference into the answer? AtP screen + real patching check.

A neuron = one coordinate of the input to mlp.down_proj (act(gate) * up), the same definition as
Neuron-Microscope's "product" channel, addressed as L<layer>_N<index> (both zero-based).

Per case x form, P and N run as one token-aligned batch, with one backward pass of m_P + m_N:
    a      activation of each neuron at each position (P and N rows)
    g      dm/da in the same run (the two rows do not interact)
Per (segment, layer, neuron), summed over the segment's positions:
    delta      mean(a_P) - mean(a_N); z = delta / neuron std over all prompt tokens     (reference)
    atp        sum((a_P - a_N) * g)
               g from the N run = "denoise" (predicted m_patched - m_N when N gets P's value)
               g from the P run = "noise"   (predicted m_P - m_patched when P gets N's value)
    Both estimate (m shift toward the P side); atp / (m_P - m_N) = predicted patching recovery on
    the scale of patch_posneg.py. Pairs with m_P - m_N < --min-gap are not ranked (unstable).

Ranking uses the two directions' average (the trapezoid of a_N..a_P, closer to real patching than
either end). Both directions and their difference are reported. Sign: + = the neuron's P-N
difference pushes the answer toward the P side; - = it pushes against.
Cells = test x form (12 in scope "all"; 4 per single form); a cell is its mean predicted recovery.
    consistent   smallest same-signed cell value (0 if any cell disagrees): recurs everywhere
    largest      |mean of the cell values|
    delta        (diff segment only, kept next to AtP) |mean of the cells' mean z|: the neurons whose
                 activation differs most at the differing word, in scope all and per single form

Guards: within each test, patients alternate into half A (select) and half B (never used to select);
--holdout-tests go entirely to B. All wordings and forms of a patient stay in one half.
null_p95: flip the sign per patient in half A, take the segment's top score, 95th percentile over
flips. A sign flip tests "effect is symmetric around 0 across patients".

Real patching check on half B, default light (--check-pairs random pairs; AtP top k combined, diff AtP
and diff delta groups, their random controls, ceiling); --full-check = every group and pair.
Both directions, one forward per row, no addition of effects: overwrite only the chosen neurons at
the segment's positions with the other run's values.
    top k       top k sites (segment, layer, neuron) from half A, patched together
    random k    same segment and layer as the top k, random neuron index
    ceiling     every neuron at the segment's positions at every layer
    single      each of the top --single-top sites alone
Compares predicted and actual recovery.

Output: results/neuron/<model_tag>/
    top_neurons.csv    per scope x segment x ranking
    candidates.txt     ids: scope all, consistent, above null_p95, all half-B cells agree
    candidates_diff_delta.txt  ids: delta ranking at diff, per scope, above null_p95, half-B cells agree
    patch_check.jsonl  real patching (recovery, pred_recovery); patch_summary.csv means
    summary.json       thresholds, |delta|-vs-atp overlap, faithfulness; pairs.jsonl; run_config.json
    attr.npy (float16 [pair, segment, layer, neuron, denoise/noise]), deltas.npy, scales.npy

Run (repo root):
    python -u neuron_posneg.py --model Qwen/Qwen3-14B                     # Lambda, inside tmux
    python neuron_posneg.py --model Qwen/Qwen3-0.6B --patients-per-test 2 --n-flips 20 --device cpu
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

RANKINGS = ["consistent", "largest"]
DIRECTIONS = ["denoise", "noise"]


def capture(blocks, store, grad):
    """Keep each layer's down_proj input in store[layer]; returns hook handles."""
    def make(l):
        def hook(module, args):
            if grad:
                args[0].retain_grad()
                store[l] = args[0]
            else:
                store[l] = args[0].detach()
        return hook
    return [b.mlp.down_proj.register_forward_pre_hook(make(l)) for l, b in enumerate(blocks)]


def score_cells(cell):
    """cell: [cells, ..., neurons] -> rank scores, sign of the mean effect."""
    grand = cell.mean(0)
    sign = grand.sign()
    return {"consistent": (sign * cell).min(0).values.clamp(min=0), "largest": grand.abs()}, sign


def cell_weights(pairs, idx, cell_keys):
    """[cells, len(idx)] averaging weights; a cell with no valid pair is dropped."""
    w = torch.zeros(len(cell_keys), len(idx))
    for j, i in enumerate(idx):
        w[cell_keys.index((pairs[i]["test_id"], pairs[i]["form"])), j] = 1
    n = w.sum(1)
    return w[n > 0] / n[n > 0, None], [k for k, c in zip(cell_keys, n.tolist()) if c > 0]


def top_sites(score, k):
    """Sites with score > 0, best first: [(value, layer, neuron)]."""
    v, i = score.flatten().topk(min(k, score.numel()))
    out = []
    for x, j in zip(v.tolist(), i.tolist()):
        if x > 0:
            out.append((x, *map(int, np.unravel_index(j, score.shape))))
    return out


@torch.no_grad()
def run_rows(hf, blocks, ids, src, rows, seg, batch, letter_ids, ti, device):
    """Each row: dict(sites=[(segment, layer, neuron)], full=[segments] or None). Returns m per row."""
    out = []
    n_layers = len(blocks)
    for start in range(0, len(rows), batch):
        chunk = rows[start: start + batch]
        idx = defaultdict(lambda: ([], [], []))
        full = defaultdict(list)
        for r, row in enumerate(chunk):
            for s, l, n in row["sites"]:
                for p in seg.get(s, []):
                    a, b, c = idx[l]
                    a.append(r), b.append(p), c.append(n)
            if row["full"]:
                pos = [p for s in row["full"] for p in seg.get(s, [])]
                for l in range(n_layers):
                    full[l].append((r, pos))

        def make(l):
            ix = [torch.tensor(v, device=device) for v in idx[l]] if l in idx else None
            def hook(module, args):
                x = args[0].clone()
                if ix is not None:
                    x[ix[0], ix[1], ix[2]] = src[l][ix[1], ix[2]]
                for r, pos in full.get(l, []):
                    x[r, pos] = src[l][pos]
                return (x,)
            return hook

        handles = [b.mlp.down_proj.register_forward_pre_hook(make(l))
                   for l, b in enumerate(blocks) if l in idx or l in full]
        try:
            batch_ids = torch.tensor([ids] * len(chunk), device=device)
            logits = hf(input_ids=batch_ids, logits_to_keep=1, use_cache=False).logits[:, -1, :]
        finally:
            for h in handles:
                h.remove()
        out += metric(logits, letter_ids, ti).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--pool", default="data/posneg_pool_v2.jsonl")
    ap.add_argument("--out-root", default="results/neuron")
    ap.add_argument("--patients-per-test", type=int, default=10)
    ap.add_argument("--holdout-tests", nargs="*", default=["amylase", "us_appendix"])
    ap.add_argument("--check-pairs", type=int, default=50, help="pairs in the light patching check")
    ap.add_argument("--full-check", action="store_true", help="every group on every half-B pair")
    ap.add_argument("--min-gap", type=float, default=1.0)
    ap.add_argument("--top", type=int, default=200, help="neurons listed per scope x segment x ranking")
    ap.add_argument("--n-flips", type=int, default=200)
    ap.add_argument("--ks", type=int, nargs="+", default=[50])
    ap.add_argument("--single-top", type=int, default=0)
    ap.add_argument("--patch-batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    out = Path(args.out_root) / args.model.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    for p in hf.parameters():
        p.requires_grad_(False)
    blocks = hf.model.layers
    n_layers, n_neurons = len(blocks), blocks[0].mlp.down_proj.in_features
    letter_ids = tok.convert_tokens_to_ids(LETTERS)

    per_test = {}
    for line in open(args.pool):
        c = json.loads(line)
        per_test.setdefault(c["test_id"], []).append(c)
    assert args.patients_per_test >= 2, "need at least one patient in each half"
    tests = list(per_test)
    pool = []
    for t in tests:                                         # a patient's records all go to one half
        patients = list(dict.fromkeys(c["hadm_id"] for c in per_test[t]))[: args.patients_per_test]
        halves = {h: "B" if t in args.holdout_tests or i % 2 else "A" for i, h in enumerate(patients)}
        pool += [(c, halves[c["hadm_id"]]) for c in per_test[t] if c["hadm_id"] in halves]

    # ---- stage 1: activations and gradients ----------------------------------------
    n_pairs = len(pool) * len(FORMS)
    shape = (n_pairs, len(SEGMENTS), n_layers, n_neurons)
    delta = torch.full(shape, float("nan"), dtype=torch.float32)
    attr = torch.full((*shape, 2), float("nan"), dtype=torch.float32)
    total = torch.zeros(n_layers, n_neurons, dtype=torch.float64, device=device)
    total2 = total.clone()
    count = 0
    pairs, skipped, leak = [], 0, 0.0
    store = {}
    handles = capture(blocks, store, grad=True)
    embed = hf.get_input_embeddings()

    t0 = time.time()
    for done, (c, half) in enumerate(pool, 1):
        ti = LETTERS.index(c["target_letter"])
        for form in FORMS:
            fm = c["forms"][form]
            _, p_ids, offs = encode(tok, fm["P"])
            _, n_ids, _ = encode(tok, fm["N"])
            if len(p_ids) != len(n_ids):                    # pool was aligned with another tokenizer
                skipped += 1
                pairs.append(None)
                continue
            seg = segments(tok, fm["P"], fm["P_result"], p_ids, n_ids, offs)
            pi = len(pairs)
            inputs = embed(torch.tensor([p_ids, n_ids], device=device)).detach().requires_grad_(True)
            logits = hf(inputs_embeds=inputs, logits_to_keep=1, use_cache=False).logits[:, -1, :]
            m = metric(logits, letter_ids, ti)
            m.sum().backward()
            pairs.append(dict(case_id=c["case_id"], hadm_id=c["hadm_id"], test_id=c["test_id"], form=form,
                              wording=c.get("wording", 0), half=half, target=c["target_letter"],
                              m_P=m[0].item(), m_N=m[1].item()))
            first = seg["diff"][0]
            for l in range(n_layers):
                a, g = store[l].detach().float(), store[l].grad.float()      # [2, T, N]
                body = a[:, 1:].double()                                     # position 0 is a "sink" token
                total[l] += body.sum((0, 1))
                total2[l] += body.pow(2).sum((0, 1))
                if l == 0:
                    count += body.shape[0] * body.shape[1]
                if first > 1:                                                # identical prefix: P == N
                    leak = max(leak, (a[0, 1:first] - a[1, 1:first]).abs().max().item())
                d = a[0] - a[1]
                for si, name in enumerate(SEGMENTS):
                    if name in seg:
                        pos = torch.tensor(seg[name], device=device)
                        delta[pi, si, l] = d[pos].mean(0).cpu()
                        attr[pi, si, l, :, 0] = (d[pos] * g[1, pos]).sum(0).cpu()
                        attr[pi, si, l, :, 1] = (d[pos] * g[0, pos]).sum(0).cpu()
            store.clear()
        print(f"  gradients {done}/{len(pool)} cases ({time.time() - t0:.0f}s)", flush=True)
    for h in handles:
        h.remove()

    scale = (total2 / count - (total / count).pow(2)).clamp(min=0).sqrt().float().cpu()
    scale = torch.maximum(scale, (1e-2 * scale.median(1, keepdim=True).values).clamp(min=1e-6))
    z = delta / scale
    gaps = torch.tensor([p["m_P"] - p["m_N"] if p else float("nan") for p in pairs])
    print(f"stage 1 done ({time.time() - t0:.0f}s); {skipped} pairs skipped (not token-aligned); "
          f"{int((gaps >= args.min_gap).sum())}/{n_pairs} pairs have gap >= {args.min_gap}; "
          f"max |P-N| before the first differing token = {leak:.4f} (should be ~0)", flush=True)

    with (out / "pairs.jsonl").open("w") as f:
        for p in pairs:
            if p:
                f.write(json.dumps(p) + "\n")
    np.save(out / "attr.npy", attr.to(torch.float16).numpy())
    np.save(out / "deltas.npy", delta.to(torch.float16).numpy())
    np.save(out / "scales.npy", scale.numpy())

    # ---- stage 2: rank on half A, check on half B ----------------------------------
    scopes = {"all": FORMS, **{f: [f] for f in FORMS}}
    all_cells = [(t, f) for f in FORMS for t in tests]
    patients = sorted({p["hadm_id"] for p in pairs if p})
    gen = torch.Generator().manual_seed(args.seed)
    flips = torch.randint(0, 2, (args.n_flips, len(patients)), generator=gen).float() * 2 - 1
    rows, summary, chosen, diff_groups = [], {}, {}, {}
    for scope, forms in scopes.items():
        keys = [k for k in all_cells if k[1] in forms]
        for si, name in enumerate(SEGMENTS):
            valid = [i for i, p in enumerate(pairs)
                     if p and p["form"] in forms and gaps[i] >= args.min_gap and not attr[i, si, 0, 0, 0].isnan()]
            ia = [i for i in valid if pairs[i]["half"] == "A"]
            ib = [i for i in valid if pairs[i]["half"] == "B"]
            if not ia or not ib:
                continue

            def recovery(idx):                                       # [pairs, layers, neurons, directions]
                return attr[idx, si] / gaps[idx][:, None, None, None]

            wa, _ = cell_weights(pairs, ia, keys)
            wb, cells_b = cell_weights(pairs, ib, keys)
            wall, cells_all = cell_weights(pairs, valid, keys)
            ra = recovery(ia).mean(-1).to(device)
            ca = torch.einsum("cp,pln->cln", wa.to(device), ra)
            sa, sign = score_cells(ca)
            ranks = RANKINGS + ["delta"] if name == "diff" else RANKINGS
            z_a = z[ia, si].to(device)
            z_grand = torch.einsum("cp,pln->cln", wa.to(device), z_a).mean(0)
            if name == "diff":
                sa["delta"] = z_grand.abs()
            sign_z = z_grand.sign().cpu()

            nulls = {r: [] for r in ranks}
            case_a = torch.tensor([patients.index(pairs[i]["hadm_id"]) for i in ia], device=device)
            for flip in flips.to(device):
                sn, _ = score_cells(torch.einsum("cp,pln->cln", wa.to(device), flip[case_a][:, None, None] * ra))
                for r in RANKINGS:
                    nulls[r].append(sn[r].max().item())
                if name == "diff":
                    flipped = torch.einsum("cp,pln->cln", wa.to(device), flip[case_a][:, None, None] * z_a)
                    nulls["delta"].append(flipped.mean(0).abs().max().item())
            p95 = {r: float(np.percentile(nulls[r], 95)) for r in ranks}

            cb = torch.einsum("cp,pln->cln", wb, recovery(ib).mean(-1))
            cb_z = torch.einsum("cp,pln->cln", wb, z[ib, si])
            cfull = torch.einsum("cp,pln->cln", wall, recovery(valid).mean(-1)).numpy()
            mean_dir = recovery(valid).mean(0)
            mean_z = z[valid, si].mean(0)
            sign_cpu = sign.cpu()

            # |activation difference| ranking, for reference: how much does it overlap with AtP?
            za = torch.einsum("cp,pln->cln", wa, z[ia, si]).mean(0).abs()
            k = min(args.top, za.numel())
            by_delta, by_atp = set(za.flatten().topk(k).indices.tolist()), set(sa["largest"].cpu().flatten().topk(k).indices.tolist())

            for r in ranks:
                sites = top_sites(sa[r].cpu(), max(args.top, *args.ks))
                if scope == "all":
                    chosen[(r, name)] = sites
                if name == "diff" and r in ("largest", "delta"):
                    diff_groups[(scope, r)] = sites
                for rank, (v, l, n) in enumerate(sites[: args.top], 1):
                    is_delta = r == "delta"
                    s = (sign_z if is_delta else sign_cpu)[l, n].item()
                    b = s * (cb_z if is_delta else cb)[:, l, n]
                    label = ("P>N" if s > 0 else "N>P") if is_delta else ("toward P" if s > 0 else "against P")
                    rows.append(dict(
                        scope=scope, segment=name, ranking=r, rank=rank, neuron=f"L{l}_N{n}", layer=l, index=n,
                        pushes=label, score_A=round(v, 4), above_null_p95=v > p95[r],
                        worst_cell_B=round(b.min().item(), 4), mean_cell_B=round(b.mean().item(), 4),
                        cells_agree_B=int((b > 0).sum()), n_cells_B=len(cells_b),
                        pred_denoise=round(mean_dir[l, n, 0].item(), 5), pred_noise=round(mean_dir[l, n, 1].item(), 5),
                        denoise_minus_noise=round((mean_dir[l, n, 0] - mean_dir[l, n, 1]).item(), 5),
                        mean_z_delta=round(mean_z[l, n].item(), 3),
                        **{f"pred_{t}_{f}": round(float(cfull[cells_all.index((t, f)), l, n]), 4) for t, f in cells_all}))
            summary[f"{scope}/{name}"] = {
                "n_A": len(ia), "n_B": len(ib), f"overlap_top{k}_delta_vs_atp": len(by_delta & by_atp) / k,
                **{f"{r}_null_p95": p95[r] for r in ranks},
                **{f"{r}_n_above_null": int((sa[r] > p95[r]).sum()) for r in ranks},
                **{f"{r}_top": sa[r].max().item() for r in ranks}}

    columns = list(dict.fromkeys(k for row in rows for k in row))
    with (out / "top_neurons.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    keep = list(dict.fromkeys(
        r["neuron"] for r in rows if r["scope"] == "all" and r["ranking"] == "consistent"
        and r["above_null_p95"] and r["cells_agree_B"] == r["n_cells_B"] and r["worst_cell_B"] > 0))
    (out / "candidates.txt").write_text("".join(n + "\n" for n in keep))
    keep_diff = list(dict.fromkeys(
        r["neuron"] for r in rows if r["ranking"] == "delta" and r["above_null_p95"]
        and r["cells_agree_B"] == r["n_cells_B"] and r["worst_cell_B"] > 0))
    (out / "candidates_diff_delta.txt").write_text("".join(n + "\n" for n in keep_diff))
    print(f"ranking done ({time.time() - t0:.0f}s); {len(keep)} candidates", flush=True)

    # ---- stage 3: real patching on half B -------------------------------------------
    rng = np.random.default_rng(args.seed)
    combined = {r: sorted(((v, s, l, n) for (rk, s), xs in chosen.items() if rk == r for v, l, n in xs), reverse=True)
                for r in RANKINGS}
    plan = [dict(group="noop", ranking="", kind="noop", k=0, sites=[], full=None)]
    for r in RANKINGS:
        groups = {s: [(v, s, l, n) for v, l, n in xs] for (rk, s), xs in chosen.items() if rk == r}
        groups["combined"] = combined[r]
        for g, sites in groups.items():
            for k in args.ks:
                if len(sites) < k:
                    continue
                top = [(s, l, n) for _, s, l, n in sites[:k]]
                plan.append(dict(group=g, ranking=r, kind="top", k=k, sites=top, full=None))
                plan.append(dict(group=g, ranking=r, kind="random", k=k, full=None,
                                 sites=[(s, l, int(rng.integers(n_neurons))) for s, l, _ in top]))
    for (scope, r), sites in diff_groups.items():                     # diff: AtP vs activation difference
        form = None if scope == "all" else scope
        for k in args.ks:
            if len(sites) < k:
                continue
            top = [("diff", l, n) for _, l, n in sites[:k]]
            plan.append(dict(group=f"diff:{scope}", ranking=r, kind="top", k=k, sites=top, full=None, form=form))
            plan.append(dict(group=f"diff:{scope}", ranking=r, kind="random", k=k, full=None, form=form,
                             sites=[(s, l, int(rng.integers(n_neurons))) for s, l, _ in top]))
    for s in SEGMENTS:
        plan.append(dict(group=s, ranking="", kind="ceiling", k=0, sites=[], full=[s]))
    plan.append(dict(group="combined", ranking="", kind="ceiling", k=0, sites=[], full=list(SEGMENTS)))
    for i, (v, s, l, n) in enumerate(combined["largest"][: args.single_top]):
        plan.append(dict(group="combined", ranking="largest", kind="single", k=i + 1, sites=[(s, l, n)], full=None))

    if not args.full_check:
        plan = [r for r in plan if r["kind"] == "noop" or (r["kind"] == "ceiling" and r["group"] == "combined")
                or (r["kind"] in ("top", "random")
                    and ((r["group"] == "combined" and r["ranking"] == "largest") or r["group"].startswith("diff:")))]

    results, noop_err = [], 0.0
    t1 = time.time()
    check = [i for i, p in enumerate(pairs) if p and p["half"] == "B" and gaps[i] >= args.min_gap]
    if not args.full_check and len(check) > args.check_pairs:
        check = sorted(rng.choice(check, args.check_pairs, replace=False).tolist())
    by_id = {c["case_id"]: c for c, _ in pool}
    clean_store = {}
    with (out / "patch_check.jsonl").open("w") as f:
        for done, pi in enumerate(check, 1):
            p = pairs[pi]
            c = by_id[p["case_id"]]
            ti = LETTERS.index(c["target_letter"])
            fm = c["forms"][p["form"]]
            _, p_ids, offs = encode(tok, fm["P"])
            _, n_ids, _ = encode(tok, fm["N"])
            seg = segments(tok, fm["P"], fm["P_result"], p_ids, n_ids, offs)
            handles = capture(blocks, clean_store, grad=False)
            with torch.no_grad():
                hf(input_ids=torch.tensor([p_ids, n_ids], device=device), logits_to_keep=1, use_cache=False)
            for h in handles:
                h.remove()
            src = {"P": {l: v[0] for l, v in clean_store.items()}, "N": {l: v[1] for l, v in clean_store.items()}}
            gap = p["m_P"] - p["m_N"]
            for di, direction in enumerate(DIRECTIONS):
                host, other = (n_ids, "P") if direction == "denoise" else (p_ids, "N")
                here = [row for row in plan if row.get("form") in (None, p["form"])]
                ms = run_rows(hf, blocks, host, src[other], here, seg, args.patch_batch, letter_ids, ti, device)
                base = p["m_N"] if direction == "denoise" else p["m_P"]
                noop_err = max(noop_err, abs(ms[0] - base) / max(1.0, abs(gap)))
                for row, mp in zip(here, ms):
                    if row["kind"] == "noop":
                        continue
                    if row["full"]:
                        pred = sum(attr[pi, SEGMENTS.index(s), :, :, di].nansum().item() for s in row["full"]
                                   if s in seg) / gap
                    else:
                        pred = sum(attr[pi, SEGMENTS.index(s), l, n, di].item() for s, l, n in row["sites"]
                                   if s in seg) / gap
                    rec = (mp - p["m_N"]) / gap if direction == "denoise" else (p["m_P"] - mp) / gap
                    record = dict(case_id=p["case_id"], test_id=p["test_id"], form=p["form"], direction=direction,
                                  group=row["group"], ranking=row["ranking"], kind=row["kind"], k=row["k"],
                                  gap=gap, recovery=rec, pred_recovery=pred)
                    results.append(record)
                    f.write(json.dumps(record) + "\n")
            print(f"  patching {done}/{len(check)} pairs ({time.time() - t1:.0f}s)", flush=True)

    aggregate = defaultdict(list)
    for r in results:
        if r["kind"] != "single":
            aggregate[(r["group"], r["ranking"], r["kind"], r["k"], r["direction"])].append((r["recovery"], r["pred_recovery"]))
    with (out / "patch_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "ranking", "kind", "k", "direction", "n", "mean_recovery", "median_recovery", "mean_pred"])
        for key, v in sorted(aggregate.items(), key=lambda kv: str(kv[0])):
            a = np.array(v)
            writer.writerow([*key, len(v), round(a[:, 0].mean(), 4), round(float(np.median(a[:, 0])), 4), round(a[:, 1].mean(), 4)])

    faith = {}
    for direction in DIRECTIONS:
        for kinds, label in ((["single"], "single"), (["top"], "top_k")):
            a = np.array([[r["pred_recovery"], r["recovery"]] for r in results
                          if r["kind"] in kinds and r["direction"] == direction])
            if len(a) > 2 and a.std(0).min() > 0:
                faith[f"{label}/{direction}"] = {"n": len(a), "pearson": float(np.corrcoef(a.T)[0, 1]),
                                                 "mean_pred": float(a[:, 0].mean()), "mean_actual": float(a[:, 1].mean())}
    summary["faithfulness_pred_vs_actual"] = faith
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "run_config.json").write_text(json.dumps(
        dict(vars(args), device=device, dtype=str(dtype), n_layers=n_layers, n_neurons=n_neurons, segments=SEGMENTS,
             max_pre_diff_leak=leak, max_noop_error=noop_err, skipped_pairs=skipped, n_cases=len(pool)), indent=2))

    print(f"noop rows differ from clean by at most {noop_err:.4f} of the gap (should be ~0)")
    for key, v in faith.items():
        print(f"AtP vs real patching {key:18s} n={v['n']:5d} r={v['pearson']:.2f} "
              f"mean pred {v['mean_pred']:.3f} actual {v['mean_actual']:.3f}")
    for key, v in sorted(aggregate.items(), key=lambda kv: str(kv[0])):
        if (key[0] == "combined" and key[2] in ("top", "random", "ceiling")) or (key[0].startswith("diff:") and key[2] == "top"):
            print(f"{key[0]:14s} {key[1]:11s} {key[2]:8s} k={key[3]:<4d} {key[4]:8s} n={len(v):4d} "
                  f"mean recovery {np.mean([x[0] for x in v]):.3f}")
    print(f"done ({time.time() - t0:.0f}s) -> {out}", flush=True)


if __name__ == "__main__":
    main()
