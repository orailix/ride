#!/bin/bash
#SBATCH --account=<slurm-account>
#SBATCH --job-name=optuna-transformer-lite
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --hint=nomultithread
#SBATCH -C v100-32g

set -euo pipefail

source ~/.bashrc
conda activate ride

cd /path/to/ride

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

DATA_DIR="data/gold/lite/tabular"
OUTPUT_DIR="runs/benchmark/lite/optuna_transformer"
STUDY_NAME="optuna_transformer_lite"
SEARCH_SPACE_CONFIG="configs/benchmark/optuna/lite/optuna_transformer.yaml"

TIME_BUDGET_HOURS="19"
MAX_TRIALS="50"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-}"

srun python -u -m scripts.benchmark.optuna.transformer \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --search-space-config "$SEARCH_SPACE_CONFIG" \
  --study-name "$STUDY_NAME" \
  --time-budget-hours "$TIME_BUDGET_HOURS" \
  --max-trials "$MAX_TRIALS"
