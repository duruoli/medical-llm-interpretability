"""Build a CLEAN working copy for a blind repair round on a new model (PIPELINE.md stage 2, blind phase).

The researcher agent in a blind round must not see what earlier models taught us (Qwen layers, heads,
read vectors, round reports). The repo is full of that (HANDOFF_round*.md, PIPELINE.md, PLAN_*.md, config
comments), so the agent works in a separate directory that holds only:
  harness/{__init__,client,core,server}.py   code, module docstrings replaced by neutral ones
  harness/config.py                          rebuilt from the real config: only the names the server needs,
                                             no comments, LIT trimmed to the practice tests, one free arm
  data/harness/practice.jsonl                the suite's practice split (MIMIC text; chmod 700)
  HANDOFF.md                                 the blind brief
  .claude/settings.json                      deny rules for the other project directories
It then greps the copy for leak words and refuses to finish if any are found.

  python blind_setup.py --suite repair_llama --model meta-llama/Llama-3.1-8B-Instruct \
      --brief HANDOFF_blind_llama.md --dest /projects/p53430/medllm/llama_blind
Run from the repo root on Quest, after `python -m harness.make_splits --suite repair_llama`.
"""
import argparse
import ast
import json
import os
import re
import shutil
import sys
from pathlib import Path

from harness.config import SUITES

# names harness/core.py and harness/server.py import from config (+ what arm_check / lit_moves use)
KEEP = {"lit_moves", "PB_RANGE", "DOWN_FLOOR", "FORBIDDEN", "MAX_NNZ", "BUDGET", "MAX_PROBE_PER_CALL",
        "GENERAL_YESNO", "GENERAL_SUFFIX", "NLL_TEXTS", "NLL_MAX_TOKENS", "arm_check"}
LEAK = re.compile(r"Qwen|DeltaNet|linear_attn|27\.1[0-3]|31\.5|\bdV\b|\bC3\b|\bround[ -]?[1-6]\b|HANDOFF_|"
                  r"B_translate|A_readout|ridge|N_off|P_off", re.I)
NEUTRAL_DOC = {
    "core.py": '"""Model wrapper, weight editing with rollback, diagnostic interventions, scoring and guards."""\n',
    "server.py": '"""GPU service: holds the model, enforces the rules and the budget. See HANDOFF.md."""\n',
    "client.py": '"""Command-line client for harness.server. See HANDOFF.md."""\n',
}


def strip_docstring(src, new_doc):
    tree = ast.parse(src)
    first = tree.body[0]
    if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
            and isinstance(first.value.value, str):
        lines = src.splitlines(keepends=True)
        return new_doc + "".join(lines[first.end_lineno:])
    return new_doc + src


def build_config(practice_tests, n_layers, budget):
    src = Path("harness/config.py").read_text()
    tree = ast.parse(src)
    body = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in KEEP:
            node.body = [s for s in node.body
                         if not (isinstance(s, ast.Expr) and isinstance(getattr(s, "value", None), ast.Constant))]
            body.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in KEEP for t in node.targets):
            if any(t.id == "BUDGET" for t in node.targets if isinstance(t, ast.Name)):
                continue                       # written below from --score-calls / --probe-prompts
            body.append(node)
    out = ['"""Fixed settings of the harness (yardsticks, rules, budget). Read-only for the researcher."""']
    out += [ast.unparse(n) for n in body]
    import harness.config as C
    lit = {t: C.LIT[t] for t in practice_tests}
    out.append(f"LIT = {lit!r}")
    out.append(f"BUDGET = {budget!r}")
    out.append(f"ARMS = {{'free': {{'layers': range(0, {n_layers}), 'modules': ('self_attn', 'mlp')}}}}")
    out.append("ARM_PARAM = re.compile('layers\\\\.(\\\\d+)\\\\.(self_attn|mlp)\\\\.')")
    return "\n\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=list(SUITES))
    ap.add_argument("--model", required=True)
    ap.add_argument("--brief", required=True, help="the blind HANDOFF file (copied as HANDOFF.md)")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--n-layers", type=int, default=32)
    ap.add_argument("--score-calls", type=int, default=100)
    ap.add_argument("--probe-prompts", type=int, default=8000)
    ap.add_argument("--deny", nargs="*", default=["/home/dlf8982/AAA", "/scratch/dlf8982",
                                                  "/projects/p53430/medllm/harness*",
                                                  "/projects/p53430/medllm/diagnose"],
                    help="directories the agent may not read (earlier rounds and the full repo)")
    args = ap.parse_args()

    suite = SUITES[args.suite]
    practice = Path(suite["practice"])
    if not practice.exists():
        sys.exit(f"{practice} missing: run `python -m harness.make_splits --suite {args.suite}` first")
    tests = list(json.loads(practice.open().readline())["tests"])

    dest = Path(args.dest)
    repo = dest / "repo"
    if repo.exists():
        sys.exit(f"{repo} exists; remove it first (a blind copy is built once)")
    (repo / "harness").mkdir(parents=True)
    (repo / "data" / "harness").mkdir(parents=True)
    (repo / ".claude").mkdir()
    (dest / "out").mkdir(exist_ok=True)

    (repo / "harness" / "__init__.py").write_text("")
    for f in ("core.py", "server.py", "client.py"):
        src = Path("harness", f).read_text()
        src = strip_docstring(src, NEUTRAL_DOC[f])
        src = src.replace('default="Qwen/Qwen3.5-9B"', f'default="{args.model}"')
        src = src.replace('Path("/scratch/dlf8982/harness")', f'Path("{dest / "out"}")')
        src = src.replace('default="/projects/p53430/medllm/harness"', f'default="{dest / "out"}"')
        (repo / "harness" / f).write_text(src)
    (repo / "harness" / "config.py").write_text(build_config(tests, args.n_layers, {"score_calls": args.score_calls, "probe_prompts": args.probe_prompts}))
    shutil.copy(practice, repo / "data" / "harness" / "practice.jsonl")
    shutil.copy(args.brief, repo / "HANDOFF.md")

    deny = []
    for d in args.deny:
        deny += [f"Read(/{d}/**)", f"Edit(/{d}/**)", f"Grep(/{d}/**)", f"Glob(/{d}/**)"]
    deny += [f"Edit(/{repo}/harness/**)", f"Edit(/{dest}/out/budget.json)", f"Edit(/{dest}/out/log.jsonl)"]
    (repo / ".claude" / "settings.json").write_text(json.dumps({"permissions": {"deny": deny}}, indent=1))

    # leak check on everything the agent can read in the copy (the practice data is MIMIC text, skipped)
    hits = []
    for p in repo.rglob("*"):
        if p.is_file() and p.suffix in (".py", ".md", ".json") and p.name != "settings.json":
            for i, line in enumerate(p.read_text().splitlines(), 1):
                if LEAK.search(line):
                    hits.append(f"{p.relative_to(repo)}:{i}: {line.strip()[:120]}")
    if hits:
        print("LEAK CHECK FAILED (fix the brief or this script, delete the copy, rerun):")
        print("\n".join(hits))
        sys.exit(1)

    for p in (dest, repo, dest / "out"):
        os.chmod(p, 0o700)
    print(f"clean copy ready: {repo}\n  practice: {len(tests)} tests {tests}\n  out: {dest / 'out'}\n"
          f"  leak check: clean\n  start the server FROM {repo} (python -m harness.server ...)")


if __name__ == "__main__":
    main()
