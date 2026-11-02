#!/bin/bash

# Value vs reference range: numeric result with the value fixed and the stated range changed,
# behavior only, all 40 pool cases, Qwen3.5-9B (see range_posneg.py).
# Submit from the repo root on Quest:  sbatch range_posneg.slurm
# Same env / caches as patch_posneg.slurm (README "Running on Quest").
# Inputs:  data/posneg_pool.jsonl
# Output:  results/range/Qwen3.5-9B/   (scp back to the laptop)
# Expected runtime ~5 min (40 cases x 7 forward passes + model loading); wall time 30 min.

#SBATCH --account=p31777
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1                          # 9B in bf16 ~19 GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G                                  # weights pass through CPU RAM while loading
#SBATCH --time=00:30:00
#SBATCH --job-name=range_posneg
#SBATCH --output=runs/range_posneg_%j.out
#SBATCH --error=runs/range_posneg_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=duruoli2024@u.northwestern.edu

PROJ=/projects/p31777

module purge all
source "$HOME/miniconda/bin/activate"
conda activate "$PROJ/envs/interp"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd "$SLURM_SUBMIT_DIR"                               # = repo root you ran sbatch from
mkdir -p runs results/range

echo "Job started at: $(date)   Node: $(hostname)   Job ID: $SLURM_JOB_ID"
python --version
nvidia-smi

export HF_HOME="$PROJ/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python -u range_posneg.py \
    --model Qwen/Qwen3.5-9B \
    --pool data/posneg_pool.jsonl

echo "Job completed at: $(date)"
