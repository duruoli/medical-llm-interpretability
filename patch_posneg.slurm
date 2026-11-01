#!/bin/bash

# Positive vs negative evidence: behavior + residual-stream activation patching on Qwen3.5-9B.
# Submit from the repo root on Quest:  sbatch patch_posneg.slurm
# One-time setup: see README "Running on Quest". The env, pip cache and HF cache all live
# under /projects/$PROJ, NOT $HOME (home quota ~80 GB fills up: Errno 122 Disk quota exceeded).
# Separate env because Qwen3.5 needs transformers>=5.2 and `congraph` pins an older one.
# Inputs:  data/posneg_pool.jsonl  (built locally by build_posneg_cases.py)
# Output:  results/posneg/Qwen3.5-9B/   then run plot_posneg.py on it

#SBATCH --account=p31777
#SBATCH --partition=gengpu                          # ⚠️ CONFIRM this partition has the GPU below
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=patch_posneg
#SBATCH --output=runs/patch_posneg_%j.out
#SBATCH --error=runs/patch_posneg_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=duruoli2024@u.northwestern.edu

PROJ=/projects/p31777

module purge all
source "$HOME/miniconda/bin/activate"
conda activate "$PROJ/envs/interp"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd "$SLURM_SUBMIT_DIR"                               # = repo root you ran sbatch from
mkdir -p runs results/posneg

echo "Job started at: $(date)   Node: $(hostname)   Job ID: $SLURM_JOB_ID"
python --version
nvidia-smi

export HF_HOME="$PROJ/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u patch_posneg.py \
    --model Qwen/Qwen3.5-9B \
    --pool data/posneg_pool.jsonl \
    --pool-per-test 10 --select-per-test 5

python plot_posneg.py --run results/posneg/Qwen3.5-9B

echo "Job completed at: $(date)"
