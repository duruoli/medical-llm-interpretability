"""Do late layers cancel the evidence, and more so near the floor?  (PLAN_crossmodel.md, question 3, pre-repair test)

For every patient x test: run the no-result prompt B and remember, at the answer position, what each listed
late module (MLP / attention of a layer) writes. Then run each result prompt (P, N) normally and again with
those modules forced to write what they wrote on B, i.e. blind to the result at the answer position
(earlier layers and earlier positions still see it). If a module cancels the evidence, blinding it makes the
move larger. The question is WHERE on the prior curve: if the down move at low priors grows most, the floor
has a locatable cause.

  python -m harness.cancel_bins --model Qwen/Qwen3.5-9B --out /scratch/dlf8982/cancel/Qwen3.5-9B
  python -m harness.cancel_bins --model Qwen/Qwen3.5-9B --out <same> --from-margins     # report only, no GPU

Conditions (--cond NAME=SPEC, repeatable): SPEC = comma list of m<L> / a<L> / m<L1>:<L2> (inclusive),
negative layers count from the end (m-1 = last MLP). Default: last MLP; whole last layer; last 5 MLPs.
Output <out>/margins.jsonl (one line per patient, appended as it goes; a rerun resumes), REPORT.md.
Batch 1 (exact). Default 3 wordings per test (impression/findings/lexical/numeric first).
"""
import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch

from harness.config import lit_moves
from harness.diagnose import FAMILIES, PRIOR_BIN_LABELS, PRIOR_BINS, Runner, build_items

DEFAULT_CONDS = ["mlp_last=m-1", "layer_last=a-1,m-1", "mlp_last5=m-5:-1"]
PAIR_PREF = ["impression", "findings", "lexical", "numeric", "full_impression", "new_numeric", "new_impression", "implicit"]
IMG = ["ct_sigmoid", "ct_appendix", "us_appendix", "us_gallbladder", "ct_kidney_stone"]
FAMS = {"ALL imaging": IMG, **FAMILIES}


def parse_spec(spec, n_layers):
    out = []
    for tok in spec.split(","):
        m = re.fullmatch(r"([am])(-?\d+)(?::(-?\d+))?", tok.strip())
        if not m:
            raise ValueError(f"bad module spec {tok!r}")
        a, b = int(m.group(2)), int(m.group(3) if m.group(3) is not None else m.group(2))
        a, b = a % n_layers, b % n_layers
        out += [(m.group(1), L) for L in range(a, b + 1)]
    return out


class Blind:
    """Forward hooks: capture a module's answer-position output on B, or overwrite it with that capture."""
    def __init__(self, hf, n_layers):
        cands = [n for n, _ in hf.named_modules() if re.fullmatch(r".*\.layers\.0", n) and "vision" not in n]
        self.prefix = cands[-1][:-2]
        self.hf, self.n = hf, n_layers
        self.mode, self.active, self.src, self.handles = "off", set(), {}, []

    def module_name(self, kind, L):
        layer = self.hf.get_submodule(f"{self.prefix}.{L}")
        if kind == "m":
            return f"{self.prefix}.{L}.mlp"
        for sub in ("self_attn", "linear_attn", "attn", "mixer"):
            if hasattr(layer, sub):
                return f"{self.prefix}.{L}.{sub}"
        raise ValueError(f"layer {L}: no attention submodule found")

    def install(self, names):
        for name in sorted(set(names)):
            self.handles.append(self.hf.get_submodule(name).register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def hook(mod, args, out):
            h = out[0] if isinstance(out, tuple) else out
            if self.mode == "capture":
                self.src[name] = h[:, -1].detach().clone()
            elif self.mode == "patch" and name in self.active:
                h = h.clone()
                h[:, -1] = self.src[name]
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            return None
        return hook


@torch.no_grad()
def margin(R, ids):
    x = torch.tensor([ids], device=R.device)
    try:
        lg = R.hf(input_ids=x, logits_to_keep=1).logits[0, -1].float()
    except TypeError:
        lg = R.hf(input_ids=x).logits[0, -1].float()
    y, n = R.yn
    return (lg[y] - lg[n]).item()


def pick_pairs(pairs, k):
    names = sorted(pairs, key=lambda p: PAIR_PREF.index(p) if p in PAIR_PREF else 99)
    return names[:k]


# ---- analysis ------------------------------------------------------------------------------
def bins_table(recs, cond, k):
    """Per family x prior bin (bin by BASELINE B / k): n, up/ideal, down/ideal (raw), asym."""
    out = {}
    for fam, tests in FAMS.items():
        row = []
        for lo, hi in zip(PRIOR_BINS[:-1], PRIOR_BINS[1:]):
            ups, dns = [], []
            for r in recs:
                for t in tests:
                    if t not in r["tests"]:
                        continue
                    T = r["tests"][t]
                    if not lo <= 1 / (1 + math.exp(-T["B"] / k)) < hi:
                        continue
                    lu, ld = lit_moves(t)
                    ps = [p for p in T if isinstance(T[p], dict)]
                    ups.append(np.mean([T[p][cond]["P"] - T["B"] for p in ps]) / lu)
                    dns.append(np.mean([T["B"] - T[p][cond]["N"] for p in ps]) / ld)
            ok = len(ups) >= 15
            row.append({"n": len(ups), "up": float(np.mean(ups)) if ok else None,
                        "down": float(np.mean(dns)) if ok else None,
                        "asym": math.log(max(np.mean(ups), .03) / max(np.mean(dns), .03)) if ok else None})
        out[fam] = row
    return out


def est_k(recs):
    lr = []
    for r in recs:
        for t, T in r["tests"].items():
            lu, ld = lit_moves(t)
            ps = [p for p in T if isinstance(T[p], dict)]
            u = np.mean([T[p]["base"]["P"] - T["B"] for p in ps]) / lu
            d = np.mean([T["B"] - T[p]["base"]["N"] for p in ps]) / ld
            if u > 0 and d > 0:
                lr.append(0.5 * (math.log(u) + math.log(d)))
    return math.exp(np.median(lr))


def report(recs, conds, args):
    k = est_k(recs)
    tabs = {c: bins_table(recs, c, k) for c in ["base"] + conds}
    f = lambda x, s="+.2f": "–" if x is None else format(x, s)
    L = [f"# Late-layer cancellation by prior bin: {args.model}", "",
         f"{len(recs)} patients; conditions: " + "; ".join(f"`{c}` = {args.spec[c]}" for c in conds)
         + f". Bins by the baseline no-result log-odds / k (k = {k:.2f}).", "",
         "Blinded = the listed modules write, at the answer position, what they wrote without a result. "
         "up, down = move / literature move (raw scale; 1 = literature x k for this model). "
         "A canceller: blinding raises the move. Floor mechanism: down rises MOST in the low bins and asym flattens.", ""]
    for fam in ["ALL imaging", "lab (lipase+amylase)", "CT diverticulitis", "appendicitis (CT+US)"]:
        L += [f"## {fam}", "", "| condition | " + " | ".join(PRIOR_BIN_LABELS) + " |", "|---|" + "---|" * 6]
        for c in ["base"] + conds:
            L.append(f"| {c} n | " + " | ".join(str(x["n"]) for x in tabs[c][fam]) + " |") if c == "base" else None
            L.append(f"| {c} asym | " + " | ".join(f(x["asym"]) for x in tabs[c][fam]) + " |")
            L.append(f"| {c} down | " + " | ".join(f(x["down"], ".2f") for x in tabs[c][fam]) + " |")
            L.append(f"| {c} up | " + " | ".join(f(x["up"], ".2f") for x in tabs[c][fam]) + " |")
        for c in conds:
            ratio = lambda a, b: "–" if a is None or b is None or b <= 0.05 else f"×{a / b:.2f}"
            L.append(f"| {c}: down gain | " + " | ".join(ratio(x["down"], y["down"]) for x, y in zip(tabs[c][fam], tabs["base"][fam])) + " |")
            L.append(f"| {c}: up gain | " + " | ".join(ratio(x["up"], y["up"]) for x, y in zip(tabs[c][fam], tabs["base"][fam])) + " |")
        L.append("")
    (args.out / "REPORT.md").write_text("\n".join(L) + "\n")
    (args.out / "report.json").write_text(json.dumps({"k": k, "tables": tabs, "spec": args.spec}, indent=1))
    print("\n".join(L))


# ---- run ---------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", default="data/sigmoid_pool.jsonl")
    ap.add_argument("--question", default="orig")
    ap.add_argument("--cond", action="append", default=None, help="NAME=SPEC, repeatable (see module doc)")
    ap.add_argument("--pairs", type=int, default=3, help="wordings per test (0 = all)")
    ap.add_argument("--limit-cases", type=int, default=188, help="stratified order, so a partial run is balanced")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--from-margins", action="store_true")
    args = ap.parse_args()
    args.out = Path(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    args.spec = dict(c.split("=", 1) for c in (args.cond or DEFAULT_CONDS))
    conds = list(args.spec)
    mpath = args.out / "margins.jsonl"

    if not args.from_margins:
        items = build_items(args.pool, args.limit_cases, args.question)
        done = {json.loads(l)["hadm_id"] for l in open(mpath)} if mpath.exists() else set()
        R = Runner(args.model, args.device, 1)
        nL = R.cfg.num_hidden_layers
        bl = Blind(R.hf, nL)
        mods = {c: [bl.module_name(kd, L) for kd, L in parse_spec(s, nL)] for c, s in args.spec.items()}
        bl.install([m for v in mods.values() for m in v])
        (args.out / "conditions.json").write_text(json.dumps({"model": args.model, "n_layers": nL, "spec": args.spec,
                                                              "modules": mods}, indent=1))
        print(f"{nL} layers; " + "; ".join(f"{c}: {len(v)} modules ({v[0]} ...)" for c, v in mods.items()), flush=True)
        tok = lambda t: R.tok(R.chat(t), add_special_tokens=False).input_ids
        t0, n_done = time.time(), 0
        for it in items:
            if it["hadm_id"] in done:
                continue
            rec = {"hadm_id": it["hadm_id"], "true_disease": it["true_disease"], "tests": {}}
            for test, T in it["tests"].items():
                bl.mode = "capture"
                row = {"B": margin(R, tok(T["B"]))}
                pairs = pick_pairs(T["pairs"], args.pairs) if args.pairs else list(T["pairs"])
                for p in pairs:
                    row[p] = {}
                    for side in ("P", "N"):
                        ids = tok(T["pairs"][p][side])
                        bl.mode = "off"
                        row[p].setdefault("base", {})[side] = margin(R, ids)
                        bl.mode = "patch"
                        for c in conds:
                            bl.active = set(mods[c])
                            row[p].setdefault(c, {})[side] = margin(R, ids)
                bl.mode = "off"
                rec["tests"][test] = row
            with mpath.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            n_done += 1
            el = time.time() - t0
            print(f"{n_done}/{len(items) - len(done)} patients, {el / 60:.1f} min, "
                  f"~{el / n_done * (len(items) - len(done) - n_done) / 60:.0f} min left", flush=True)
    recs = [json.loads(l) for l in open(mpath)]
    report(recs, conds, args)


if __name__ == "__main__":
    main()
