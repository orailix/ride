#!/bin/bash
#SBATCH --account=<slurm-account>
#SBATCH --job-name=optuna-gnn
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --hint=nomultithread
#SBATCH -C a100

set -euo pipefail

source ~/.bashrc
conda activate ride

cd /path/to/ride

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

DATA_DIR="data/gold/standard/gnn"
OUTPUT_DIR="runs/benchmark/standard/optuna_gnn"
STUDY_NAME="optuna_gnn_standard"
SEARCH_SPACE_CONFIG="configs/benchmark/optuna/standard/optuna_gnn.yaml"

TIME_BUDGET_HOURS="19"
MAX_TRIALS="50"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-}"

srun python -u -m scripts.benchmark.optuna.gnn \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --search-space-config "$SEARCH_SPACE_CONFIG" \
  --study-name "$STUDY_NAME" \
  --time-budget-hours "$TIME_BUDGET_HOURS" \
  --max-trials "$MAX_TRIALS"
