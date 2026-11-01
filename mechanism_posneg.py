"""Mechanism follow-up to patch_posneg.py: HOW does the P/N information leave the
differing token, and why do the two patching directions differ at layers ~5-11?

Runs on a few hand-picked cases (default: the 5 with the clearest noise-vs-denoise
asymmetry in the first run), all three surface forms. Three measurements:

A. components.jsonl - component patching at layers --layers (default 4..12).
   Same idea as residual patching, but only ONE part of a layer is transplanted:
     resid : the whole residual stream after the layer (reference; = patch_posneg.py)
     mixer : the token-mixing part's output. Gated DeltaNet ("linear_attention") on most
             layers, standard attention on every 4th ("full_attention").
     mlp   : the MLP's output (per-position processing, no token mixing)
   Patched positions: `diff` (the differing token) and `result_tail` (the words after it,
   e.g. the disease name). A mixer at result_tail with high recovery = that layer's mixer
   WROTE the information into those positions (it carried it over from diff).
   recovery as in patch_posneg.py (fraction of the P-N gap moved), both directions.

B. attention.jsonl - attention weights TO the differing token, at every full-attention
   layer, per head, per query segment (mean over the segment's positions; summed over the
   diff positions). Four conditions:
     clean_P, clean_N  : normal runs
     denoise_patched   : N run (host) with P's residual at diff from layer --attn-patch-layer
     noise_patched     : P run (host) with N's residual at diff from that layer
   If attention to diff follows the HOST (denoise_patched ~ clean_N), whether the position
   is read is decided by the rest of the prompt. If it follows the CONTENT
   (noise_patched ~ clean_N), the notes placed at diff decide whether they get read.

C. distance.jsonl - how different P's and N's residual streams are, per layer x segment
   (clean runs). rel_dist = |h_P - h_N| / mean(|h_P|, |h_N|), averaged over positions;
   cos = cosine similarity. Patching says whether information is USED; this says whether
   it is THERE.

Run (repo root):
  python mechanism_posneg.py --model Qwen/Qwen3.5-9B                     # Quest, ~10 min
  python mechanism_posneg.py --model Qwen/Qwen3.5-0.8B --cases lipase__28842673 --forms negation
                                                                         # local smoke test
Output: results/mech/<model_tag>/{components,attention,distance}.jsonl + run_config.json
"""
import argparse
import json
import time
from pathlib import Path

import torch
from nnsight import LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from patch_posneg import FORMS, LETTERS, encode, metric, segments

DEFAULT_CASES = ["ct_sigmoid__21055588", "ct_appendix__27541971", "us_gallbladder__23649113",
                 "us_gallbladder__20410636", "lipase__28842673"]
COMPONENTS = ["resid", "mixer", "mlp"]
PATCH_SEGS = ["diff", "result_tail"]


def mixer(lm, layer_types, l):
    lay = lm.model.layers[l]
    return lay.self_attn if layer_types[l] == "full_attention" else lay.linear_attn


def comp_out(lm, layer_types, comp, l):
    """The nnsight proxy holding the component's output tensor [batch, seq, hidden]."""
    if comp == "resid":
        return lm.model.layers[l].output
    if comp == "mlp":
        return lm.model.layers[l].mlp.output
    out = mixer(lm, layer_types, l).output
    return out[0] if layer_types[l] == "full_attention" else out


@torch.no_grad()
def clean_run(lm, ids, n_layers, layer_types, layers, diff):
    """Cache resid (all layers), mixer + mlp outputs (patch layers), and attention to diff
    at every full-attention layer. Modules are touched in execution order."""
    full = [l for l in range(n_layers) if layer_types[l] == "full_attention"]
    saved = {"resid": {}, "mixer": {}, "mlp": {}, "attn": {}}
    with lm.trace(torch.tensor([ids]), logits_to_keep=1):
        for l in range(n_layers):
            if l in layers or l in full:
                mo = mixer(lm, layer_types, l).output
                if layer_types[l] == "full_attention":
                    saved["attn"][l] = mo[1][0, :, :, diff].sum(-1).save()     # [heads, seq]
                    if l in layers:
                        saved["mixer"][l] = mo[0][0].save()
                elif l in layers:
                    saved["mixer"][l] = mo[0].save()
            if l in layers:
                saved["mlp"][l] = lm.model.layers[l].mlp.output[0].save()
            saved["resid"][l] = lm.model.layers[l].output[0].save()
        logits = lm.lm_head.output[0, -1, :].save()
    return saved, logits


@torch.no_grad()
def patch_component(lm, layer_types, comp, run_ids, src, positions, layers, letter_ids, ti):
    """Batch row i = run_ids with `comp` output at `positions` in layers[i] replaced by src."""
    batch = torch.tensor([run_ids] * len(layers))
    pos = torch.tensor(positions)
    with lm.trace(batch, logits_to_keep=1):
        for i, l in enumerate(layers):
            comp_out(lm, layer_types, comp, l)[i, pos, :] = src[comp][l][pos, :]
        logits = lm.lm_head.output[:, -1, :].save()
    return metric(logits, letter_ids, ti).tolist()


@torch.no_grad()
def attn_after_patch(lm, layer_types, n_layers, run_ids, src_resid, diff, patch_layer):
    """Host run with src's residual at diff from `patch_layer`; attention to diff at the
    full-attention layers after it."""
    full = [l for l in range(patch_layer + 1, n_layers) if layer_types[l] == "full_attention"]
    pos = torch.tensor(diff)
    out = {}
    with lm.trace(torch.tensor([run_ids]), logits_to_keep=1):
        lm.model.layers[patch_layer].output[0, pos, :] = src_resid[patch_layer][pos, :]
        for l in full:
            out[l] = lm.model.layers[l].self_attn.output[1][0, :, :, diff].sum(-1).save()
    return out


def seg_means(attn_hq, seg):
    """attn_hq [heads, seq] -> {segment: [heads]} mean over the segment's query positions."""
    return {k: attn_hq[:, v].float().mean(-1).tolist() for k, v in seg.items() if k != "diff"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--pool", default="data/posneg_pool.jsonl")
    ap.add_argument("--out-root", default="results/mech")
    ap.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    ap.add_argument("--forms", nargs="+", default=FORMS)
    ap.add_argument("--layers", type=int, nargs=2, default=[4, 12], help="inclusive range")
    ap.add_argument("--attn-patch-layer", type=int, default=5)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    out = Path(args.out_root) / args.model.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    # eager attention: the only implementation that returns attention weights
    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype,
                                              attn_implementation="eager").to(device).eval()
    lm = LanguageModel(hf, tokenizer=tok)
    n_layers = hf.config.num_hidden_layers
    layer_types = hf.config.layer_types
    layers = list(range(args.layers[0], min(args.layers[1], n_layers - 1) + 1))
    letter_ids = tok.convert_tokens_to_ids(LETTERS)
    (out / "run_config.json").write_text(json.dumps(
        dict(vars(args), device=device, dtype=str(dtype), n_layers=n_layers,
             layer_types=layer_types, patch_layers=layers), indent=2))

    pool = {c["case_id"]: c for c in map(json.loads, open(args.pool))}
    f_comp = (out / "components.jsonl").open("w")
    f_attn = (out / "attention.jsonl").open("w")
    f_dist = (out / "distance.jsonl").open("w")
    t0 = time.time()
    for cid in args.cases:
        c = pool[cid]
        ti = LETTERS.index(c["target_letter"])
        for form in args.forms:
            fm = c["forms"][form]
            _, p_ids, offs = encode(tok, fm["P"])
            _, n_ids, _ = encode(tok, fm["N"])
            seg = segments(tok, fm["P"], fm["P_result"], p_ids, n_ids, offs)
            diff = seg["diff"]
            base = {"case_id": cid, "test_id": c["test_id"], "form": form}

            cp, lp = clean_run(lm, p_ids, n_layers, layer_types, layers, diff)
            cn, ln = clean_run(lm, n_ids, n_layers, layer_types, layers, diff)
            m_p, m_n = metric(lp, letter_ids, ti).item(), metric(ln, letter_ids, ti).item()
            gap = m_p - m_n

            # C. distance between P and N residual streams
            for l in range(n_layers):
                hp, hn = cp["resid"][l].float(), cn["resid"][l].float()
                for k, v in seg.items():
                    a, b = hp[v], hn[v]
                    rel = ((a - b).norm(dim=-1) / ((a.norm(dim=-1) + b.norm(dim=-1)) / 2)).mean().item()
                    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()
                    f_dist.write(json.dumps(dict(base, layer=l, segment=k, rel_dist=rel, cos=cos)) + "\n")

            # A. component patching
            for comp in COMPONENTS:
                for sname in PATCH_SEGS:
                    if sname not in seg:
                        continue
                    for direction, run_ids, src in [("denoise", n_ids, cp), ("noise", p_ids, cn)]:
                        ms = patch_component(lm, layer_types, comp, run_ids, src, seg[sname],
                                             layers, letter_ids, ti)
                        for l, mp in zip(layers, ms):
                            rec = (mp - m_n) / gap if direction == "denoise" else (m_p - mp) / gap
                            f_comp.write(json.dumps(dict(
                                base, component=comp, layer=l, layer_type=layer_types[l],
                                segment=sname, direction=direction, m_patched=mp,
                                m_P=m_p, m_N=m_n, recovery=rec)) + "\n")

            # B. attention to diff: clean runs + residual-patched runs
            conds = {"clean_P": cp["attn"], "clean_N": cn["attn"],
                     "denoise_patched": attn_after_patch(lm, layer_types, n_layers, n_ids, cp["resid"],
                                                         diff, args.attn_patch_layer),
                     "noise_patched": attn_after_patch(lm, layer_types, n_layers, p_ids, cn["resid"],
                                                       diff, args.attn_patch_layer)}
            for cond, by_layer in conds.items():
                for l, a in by_layer.items():
                    for qseg, heads in seg_means(a, seg).items():
                        for h, w in enumerate(heads):
                            f_attn.write(json.dumps(dict(base, condition=cond, layer=l, head=h,
                                                         query_segment=qseg, attn_to_diff=w)) + "\n")
            print(f"  {cid} {form}  gap {gap:.2f}  ({time.time() - t0:.0f}s)", flush=True)
    for f in (f_comp, f_attn, f_dist):
        f.close()
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
