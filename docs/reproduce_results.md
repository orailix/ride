# Reproduce the Paper Results

This page describes how to reproduce the benchmark, hyperparameter-search, and
ablation results reported in the paper from the released Gold datasets. The
benchmark commands below run the test-evaluation entry points: each
learning-based model is trained and evaluated for the requested seeds, while
the non-learning baselines are evaluated once.

The main paper table uses the Gold Standard tier. The Lite tier is useful for
the smaller benchmark and for faster checks of the same workflow. Full runs can
be expensive, especially for the GNN, Transformer, LSTM, and XGBoost models.

Run all commands from the repository root.

## Data

Download the Gold tiers used by the paper:

```bash
python -m scripts.dataset.download.download_dataset --target gold_lite
python -m scripts.dataset.download.download_dataset --target gold_standard
```

This creates the benchmark inputs under:

```text
data/gold/lite/
data/gold/standard/
```

The shared evaluation tables are:

```text
data/gold/lite/core/test_eval_table.parquet
data/gold/standard/core/test_eval_table.parquet
```

## Fixed Configurations

The learning-based benchmark runs use the fixed configurations under:

```text
configs/benchmark/best_models/lite/
configs/benchmark/best_models/standard/
```

These are the selected hyperparameters used for the paper results. The
test-evaluation scripts run with `--val-fraction 0.0`, so the full Gold
training split of the selected tier is used for each reported seed.

## Hyperparameter Search

Optuna search spaces are stored under:

```text
configs/benchmark/optuna/lite/
configs/benchmark/optuna/standard/
```

The corresponding SLURM wrappers are under:

```text
scripts/benchmark/slurm/lite/
scripts/benchmark/slurm/standard/
```

Before submitting them, edit the SLURM account, constraint, resource requests,
and `cd /path/to/ride` line for your cluster. The wrappers run multiple workers
against a shared Optuna SQLite study under
`runs/benchmark/<tier>/optuna_<model>/`.

Submit the Lite searches:

```bash
sbatch scripts/benchmark/slurm/lite/optuna_mlp.slurm
sbatch scripts/benchmark/slurm/lite/optuna_xgboost.slurm
sbatch scripts/benchmark/slurm/lite/optuna_lstm.slurm
sbatch scripts/benchmark/slurm/lite/optuna_transformer.slurm
sbatch scripts/benchmark/slurm/lite/optuna_gnn.slurm
```

Submit the Standard searches:

```bash
sbatch scripts/benchmark/slurm/standard/optuna_mlp.slurm
sbatch scripts/benchmark/slurm/standard/optuna_xgboost.slurm
sbatch scripts/benchmark/slurm/standard/optuna_lstm.slurm
sbatch scripts/benchmark/slurm/standard/optuna_transformer.slurm
sbatch scripts/benchmark/slurm/standard/optuna_gnn.slurm
```

Each trial writes its run directory and training history under the requested
Optuna output directory.

## Benchmark Test Evaluation

### Gold Standard

Run the Translation baseline on the Gold Standard dataset:

```bash
python -m scripts.benchmark.test_eval.translation \
  --test-eval-table data/gold/standard/core/test_eval_table.parquet \
  --output-dir runs/benchmark/standard/translation
```

Run the Graph-event model on the Gold Standard dataset:

```bash
python -m scripts.benchmark.test_eval.graph_event \
  --data-dir data/gold/standard/graph_event \
  --test-eval-table data/gold/standard/core/test_eval_table.parquet \
  --output-dir runs/benchmark/standard/graph_event
```

Run the MLP model on the Gold Standard dataset:

```bash
python -m scripts.benchmark.test_eval.mlp \
  --data-dir data/gold/standard/tabular \
  --test-eval-table data/gold/standard/core/test_eval_table.parquet \
  --output-dir runs/benchmark/standard/mlp \
  --tier standard \
  --config configs/benchmark/best_models/standard/mlp.yaml \
  --seeds 0,1,2,3,4,5,6,7,8,9
```

Run the XGBoost model on the Gold Standard dataset:

```bash
python -m scripts.benchmark.test_eval.xgboost \
  --data-dir data/gold/standard/tabular \
  --test-eval-table data/gold/standard/core/test_eval_table.parquet \
  --output-dir runs/benchmark/standard/xgboost \
  --tier standard \
  --config configs/benchmark/best_models/standard/xgboost.yaml \
  --seeds 0,1,2,3,4,5,6,7,8,9
```

Run the LSTM model on the Gold Standard dataset:

```bash
python -m scripts.benchmark.test_eval.lstm \
  --data-dir data/gold/standard/sequential \
  --test-eval-table data/gold/standard/core/test_eval_table.parquet \
  --output-dir runs/benchmark/standard/lstm \
  --tier standard \
  --config configs/benchmark/best_models/standard/lstm.yaml \
  --seeds 0,1,2,3,4,5,6,7,8,9
```

Run the Transformer model on the Gold Standard dataset:

```bash
python -m scripts.benchmark.test_eval.transformer \
  --data-dir data/gold/standard/tabular \
  --test-eval-table data/gold/standard/core/test_eval_table.parquet \
  --output-dir runs/benchmark/standard/transformer \
  --tier standard \
  --config configs/benchmark/best_models/standard/transformer.yaml \
  --seeds 0,1,2,3,4,5,6,7,8,9
```

Run the GNN model on the Gold Standard dataset:

```bash
python -m scripts.benchmark.test_eval.gnn \
  --data-dir data/gold/standard/gnn \
  --test-eval-table data/gold/standard/core/test_eval_table.parquet \
  --output-dir runs/benchmark/standard/gnn \
  --tier standard \
  --config configs/benchmark/best_models/standard/gnn.yaml \
  --seeds 0,1,2,3,4,5,6,7,8,9
```

### Gold Lite

Run the Translation baseline on the Gold Lite dataset:

```bash
python -m scripts.benchmark.test_eval.translation \
  --test-eval-table data/gold/lite/core/test_eval_table.parquet \
  --output-dir runs/benchmark/lite/translation
```

Run the Graph-event model on the Gold Lite dataset:

```bash
python -m scripts.benchmark.test_eval.graph_event \
  --data-dir data/gold/lite/graph_event \
  --test-eval-table data/gold/lite/core/test_eval_table.parquet \
  --output-dir runs/benchmark/lite/graph_event
```

Run the MLP model on the Gold Lite dataset:

```bash
python -m scripts.benchmark.test_eval.mlp \
  --data-dir data/gold/lite/tabular \
  --test-eval-table data/gold/lite/core/test_eval_table.parquet \
  --output-dir runs/benchmark/lite/mlp \
  --tier lite \
  --config configs/benchmark/best_models/lite/mlp.yaml \
  --seeds 0,1,2,3,4,5,6,7,8,9
```

Run the XGBoost model on the Gold Lite dataset:

```bash
python -m scripts.benchmark.test_eval.xgboost \
  --data-dir data/gold/lite/tabular \
  --test-eval-table data/gold/lite/core/test_eval_table.parquet \
  --output-dir runs/benchmark/lite/xgboost \
  --tier lite \
  --config configs/benchmark/best_models/lite/xgboost.yaml \
  --seeds 0,1,2,3,4,5,6,7,8,9
```

Run the LSTM model on the Gold Lite dataset:

```bash
python -m scripts.benchmark.test_eval.lstm \
  --data-dir data/gold/lite/sequential \
  --test-eval-table data/gold/lite/core/test_eval_table.parquet \
  --output-dir runs/benchmark/lite/lstm \
  --tier lite \
  --config configs/benchmark/best_models/lite/lstm.yaml \
  --seeds 0,1,2,3,4,5,6,7,8,9
```

Run the Transformer model on the Gold Lite dataset:

```bash
python -m scripts.benchmark.test_eval.transformer \
  --data-dir data/gold/lite/tabular \
  --test-eval-table data/gold/lite/core/test_eval_table.parquet \
  --output-dir runs/benchmark/lite/transformer \
  --tier lite \
  --config configs/benchmark/best_models/lite/transformer.yaml \
  --seeds 0,1,2,3,4,5,6,7,8,9
```

Run the GNN model on the Gold Lite dataset:

```bash
python -m scripts.benchmark.test_eval.gnn \
  --data-dir data/gold/lite/gnn \
  --test-eval-table data/gold/lite/core/test_eval_table.parquet \
  --output-dir runs/benchmark/lite/gnn \
  --tier lite \
  --config configs/benchmark/best_models/lite/gnn.yaml \
  --seeds 0,1,2,3,4,5,6,7,8,9
```

Each learning-based command creates one directory per seed, for example:

```text
runs/benchmark/standard/mlp/seed_00/
```

Each model directory also contains:

```text
test_eval_summary.yaml
```

This summary contains the mean and standard deviation of all aggregate and
breakdown metrics over the requested seeds.

## MLP Feature-Ablation Study

The feature-ablation study is run on the Gold Lite tabular dataset with the
fixed Lite MLP configuration. It has two phases: first select one epoch count
per ablated feature family on a validation split, then run test evaluation over
the selected epochs.

Select ablation number of epochs per family:

```bash
python -m scripts.benchmark.ablation.select_mlp_epochs \
  --data-dir data/gold/lite/tabular \
  --output-dir runs/ablation/mlp_lite_epoch_selection \
  --config configs/benchmark/best_models/lite/mlp.yaml \
  --seed 0 \
  --epochs 75 \
  --early-stopping-patience 15 \
  --val-fraction 0.1 \
  --families all
```

Run the ablation test evaluation:

```bash
python -m scripts.benchmark.ablation.test_eval_mlp \
  --data-dir data/gold/lite/tabular \
  --test-eval-table data/gold/lite/core/test_eval_table.parquet \
  --output-dir runs/ablation/mlp_lite_test_eval \
  --epoch-selection-summary runs/ablation/mlp_lite_epoch_selection/epoch_selection_summary.yaml \
  --config configs/benchmark/best_models/lite/mlp.yaml \
  --families all \
  --seeds 0,1,2
```

The ablation families are:

```text
train_information
snapshot_time_features
event_planned_timing
past_delay_features
event_type_features
event_node_embeddings
local_network_context
weather_features
```

Each family writes a `test_eval_summary.yaml` under:

```text
runs/ablation/mlp_lite_test_eval/<family>/
```

[Back to README](../README.md#what-do-you-want-to-do)
