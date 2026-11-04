#!/bin/bash

# Conflict states (diff vs tail, tail MLP vs mixers) observed downstream: P-ness of every
# position x layer + final-position logit lens; 5 hand-picked cases x 3 forms, Qwen3.5-9B (see conflict_posneg.py).
# Submit from the repo root on Quest:  sbatch conflict_posneg.slurm
# Same env / caches as patch_posneg.slurm (README "Running on Quest").
# Inputs:  data/posneg_pool.jsonl
# Output:  results/conflict/Qwen3.5-9B/   (scp back to the laptop)
# Expected runtime ~15-25 min; wall time 1 h leaves headroom for model loading.

#SBATCH --account=p31777
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1                          # peak ~30 GB on the 80 GB A100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=conflict_posneg
#SBATCH --output=runs/conflict_posneg_%j.out
#SBATCH --error=runs/conflict_posneg_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=duruoli2024@u.northwestern.edu

PROJ=/projects/p31777

module purge all
source "$HOME/miniconda/bin/activate"
conda activate "$PROJ/envs/interp"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd "$SLURM_SUBMIT_DIR"                               # = repo root you ran sbatch from
mkdir -p runs results/conflict

echo "Job started at: $(date)   Node: $(hostname)   Job ID: $SLURM_JOB_ID"
python --version
nvidia-smi

export HF_HOME="$PROJ/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u conflict_posneg.py \
    --model Qwen/Qwen3.5-9B \
    --pool data/posneg_pool.jsonl

echo "Job completed at: $(date)"
