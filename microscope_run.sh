#!/bin/bash
# Neuron-Microscope batch runner for Qwen3-14B on the Lambda filesystem `medllm`.
#
#   bash microscope_run.sh --act act3 --collect "L1_N2 L3_N4 ..."      # 1) collect activations (one corpus scan for all)
#   bash microscope_run.sh --act act2 --run "L23_N14720 L26_N8105"    # 2) full pipeline per neuron (sequential)
#   bash microscope_run.sh --act act2 --run "..." --interp-only --parallel 3   # explanation step only, no GPU
#
# Collection stores ONLY the listed neurons, so list every neuron you may want later in one --collect call.
# Existing archives: act (L23_N7544, 10M tokens), act2 (73 candidates, 10M tokens).
# Prints the parameters and waits 10 s before starting (Ctrl+c to abort).
set -euo pipefail

BASE=/lambda/nfs/medllm
R=$BASE/microscope
MODEL=$BASE/hf/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18
CORPUS=$BASE/corpus/pmc_patients.jsonl
TEMPLATE=$R/L23_N7544.json          # config template (OpenRouter LLM settings); neuron/paths are replaced
TOKENS=5000000
ACT="" ; COLLECT="" ; RUN="" ; INTERP_ONLY=0 ; PARALLEL=1

while [ $# -gt 0 ]; do
  case "$1" in
    --act) ACT=$2; shift 2;;
    --collect) COLLECT=$2; shift 2;;
    --collect-file) COLLECT=$(grep -v '^\s*$' "$2" | tr '\n' ' '); shift 2;;
    --run) RUN=$2; shift 2;;
    --tokens) TOKENS=$2; shift 2;;
    --interp-only) INTERP_ONLY=1; shift;;
    --parallel) PARALLEL=$2; shift 2;;
    *) echo "unknown argument $1"; exit 1;;
  esac
done
[ -n "$ACT" ] || { echo "--act <folder name under $R> is required"; exit 1; }
[ -n "$COLLECT$RUN" ] || { echo "give --collect and/or --run"; exit 1; }
A=$R/$ACT

cd $BASE/Neuron-Microscope
source ../nm-venv/bin/activate
export HF_HOME=$BASE/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export NEURON_PAPER_API_KEY_FILE=$HOME/.openrouter_key NEURON_MICROSCOPE_GPU_WORKER=1
if [ -n "$RUN" ]; then test -s $HOME/.openrouter_key || { echo "missing ~/.openrouter_key"; exit 1; }; fi

echo "=== parameters ==="
echo "activation folder : $A"
if [ -n "$COLLECT" ]; then
  echo "collect           : $(wc -w <<< "$COLLECT") neurons, $TOKENS tokens"
  [ -e "$A" ] && { echo "ERROR: $A already exists; collection needs a new folder"; exit 1; }
fi
if [ -n "$RUN" ]; then
  echo "run               : $RUN"
  if [ -z "$COLLECT" ]; then  # run-only: every neuron must already be in the archive
    python3 - "$A/neurons.json" $RUN <<'PY' || exit 1
import json, sys
have = {n["id"] for n in json.load(open(sys.argv[1]))}
missing = [n for n in sys.argv[2:] if n not in have]
if missing:
    sys.exit(f"ERROR: not collected in {sys.argv[1]}: {missing}")
PY
  fi
  echo "mode              : $([ $INTERP_ONLY = 1 ] && echo "explanation only, $PARALLEL at a time" || echo "full pipeline, one at a time")"
fi
echo "=================="
sleep 10

if [ -n "$COLLECT" ]; then
  echo "$(date -u) collection start"
  python -m language.collection --model $MODEL --corpus $CORPUS --neurons $COLLECT \
    --tokens $TOKENS --seed 42 --batch-size 32 --batch-tokens 32768 --gpus 0 --output $A
  echo "$(date -u) collection done"
fi

if [ -n "$RUN" ]; then
  run_one() {
    local N=$1
    python3 - "$N" "$A" "$TEMPLATE" "$R/$N.json" <<'PY'
import json, sys
n, act, template, out = sys.argv[1:]
c = json.load(open(template))
c.update(neuron=n, activations=act, report=f"{act}/reports/{n}")
json.dump(c, open(out, "w"), indent=2)
PY
    local extra=""; [ $INTERP_ONLY = 1 ] && extra="--stop-after interpretation"
    echo "$(date -u) $N start"
    python -m language --config $R/$N.json --output $R/run_$N $extra > $R/run_$N.log 2>&1 \
      && echo "$(date -u) $N done" || echo "$(date -u) $N FAILED (see $R/run_$N.log)"
  }
  if [ $INTERP_ONLY = 1 ]; then
    for N in $RUN; do
      run_one $N &
      while [ $(jobs -rp | wc -l) -ge $PARALLEL ]; do sleep 5; done
    done
    wait
  else
    for N in $RUN; do run_one $N; done
  fi
fi
echo "$(date -u) all finished"
