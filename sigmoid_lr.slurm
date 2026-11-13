#!/bin/bash

# Sigmoid CT vs appendix CT: positive vs negative belief moves against literature LRs,
# behavior only, 188 real histories, Qwen3.5-9B (see sigmoid_lr.py / build_sigmoid_cases.py).
# Submit from the repo root on Quest:  sbatch sigmoid_lr.slurm
# Same env / caches as patch_posneg.slurm (README "Running on Quest").
# Inputs:  data/sigmoid_pool.jsonl   (scp from the laptop first)
# Output:  results/sigmoid_lr/Qwen3.5-9B/   (scp back to the laptop)
# Expected runtime ~15 min (188 cases x 35 forward passes + model loading); wall time 30 min.

#SBATCH --account=p31777
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1                          # 9B in bf16 ~19 GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G                                  # weights pass through CPU RAM while loading
#SBATCH --time=00:30:00
#SBATCH --job-name=sigmoid_lr
#SBATCH --output=runs/sigmoid_lr_%j.out
#SBATCH --error=runs/sigmoid_lr_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=duruoli2024@u.northwestern.edu

PROJ=/projects/p31777

module purge all
source "$HOME/miniconda/bin/activate"
conda activate "$PROJ/envs/interp"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd "$SLURM_SUBMIT_DIR"                               # = repo root you ran sbatch from
mkdir -p runs results/sigmoid_lr

echo "Job started at: $(date)   Node: $(hostname)   Job ID: $SLURM_JOB_ID"
python --version
nvidia-smi

export HF_HOME="$PROJ/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python -u sigmoid_lr.py \
    --model Qwen/Qwen3.5-9B \
    --pool data/sigmoid_pool.jsonl

echo "Job completed at: $(date)"
