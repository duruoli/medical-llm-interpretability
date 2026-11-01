#!/bin/bash

# Positive vs negative evidence: behavior + residual-stream activation patching on Qwen3.5-9B.
# Submit from the repo root on Quest:  sbatch patch_posneg.slurm
# One-time setup on a login node, in a SEPARATE env `interp` (Qwen3.5 needs transformers>=5.2;
# upgrading `congraph` could break the Qwen3-30B jobs that pin an older transformers):
#   conda create -n interp python=3.10 -y && conda activate interp
#   pip install torch "transformers>=5.2" accelerate nnsight pandas matplotlib tabulate
#   huggingface-cli download Qwen/Qwen3.5-9B              # ~19 GB; HF_HUB_OFFLINE=1 below
#   optional speed-up (CUDA build, skip if it fails; the torch fallback works):
#   pip install flash-linear-attention causal-conv1d
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

module purge all
source "$HOME/miniconda/bin/activate"
conda activate interp
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd /home/dlf8982/AAA/medical-llm-interpretability
mkdir -p runs results/posneg

echo "Job started at: $(date)   Node: $(hostname)   Job ID: $SLURM_JOB_ID"
python --version
nvidia-smi

export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u patch_posneg.py \
    --model Qwen/Qwen3.5-9B \
    --pool data/posneg_pool.jsonl \
    --pool-per-test 10 --select-per-test 5

python plot_posneg.py --run results/posneg/Qwen3.5-9B

echo "Job completed at: $(date)"
