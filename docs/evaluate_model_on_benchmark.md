# Evaluate Your Model on the Benchmark

Once you have developed a new model, either on an existing Gold dataset or on
your own Gold variant, this tutorial explains how to evaluate it using the Gold
core associated with that dataset.

## Use the Associated Gold Core

Every model-specific Gold dataset is tied to a Gold core. The core defines the
test snapshots, active trains, targets, and evaluation reference table. When you
evaluate a model, use the `test_eval_table.parquet` from the same core that was
used to create the dataset.

For example:

```text
data/gold/lite/core/test_eval_table.parquet
data/gold/standard/core/test_eval_table.parquet
data/gold/my_custom_tier/core/test_eval_table.parquet
```

Do not evaluate predictions from one Gold core against the evaluation table of
another core. The key columns and targets may not refer to the same prediction
instances.

## Produce Benchmark Predictions

The shared evaluator expects a prediction table with the same key columns and
future-delay columns as the core evaluation table:

```text
ts
train_id
service_date
future_delay_1
future_delay_2
...
future_delay_n
```

The key columns identify the prediction instance. The `future_delay_*` columns
contain the model predictions in seconds. The number of future-delay columns must
match the `future_delay_*` columns in `test_eval_table.parquet`.

For an in-repository model, the usual pattern is to write an evaluation script
under `scripts/benchmark/eval/`. Existing examples include:

```text
scripts/benchmark/eval/mlp.py
scripts/benchmark/eval/xgboost.py
scripts/benchmark/eval/lstm.py
scripts/benchmark/eval/transformer.py
scripts/benchmark/eval/gnn.py
scripts/benchmark/eval/graph_event.py
```

These scripts load a trained model, run it on the test split, convert model
outputs back to absolute future-delay predictions in seconds, and write:

```text
test_predictions.parquet
eval_metrics.yaml
```

If your model is trained outside this repository, you can still use the same
evaluation protocol by exporting a prediction table with the columns above. Any
target transformation used during training must be reversed before evaluation so
that `future_delay_*` predictions are absolute delays in seconds.

## Run the Shared Evaluator

The shared metric function is:

```python
from src.benchmark.utils.evaluation import evaluate_delay_predictions
```

It joins predictions to the core evaluation table on:

```text
ts, train_id, service_date
```

Then it computes global MAE/RMSE and the standard breakdowns by prediction
horizon and delay-change bin.

A minimal standalone evaluation looks like this:

```python
from pathlib import Path

import pandas as pd

from src.benchmark.utils.evaluation import evaluate_delay_predictions
from src.dataset.pipeline.helpers import write_yaml


def load_your_model(model_path):
    pass


def load_your_test_split(data_dir):
    pass


def convert_model_outputs_to_delay_seconds(raw_outputs, metadata, eval_table):
    # Undo your model's target transforms here and return a DataFrame with:
    # ts, train_id, service_date, future_delay_1, ..., future_delay_n.
    # The future_delay_* values must be absolute delays in seconds.
    pass


eval_table = pd.read_parquet("data/gold/my_custom_tier/core/test_eval_table.parquet")
model = load_your_model("runs/my_model/model.pt")
test_inputs, metadata = load_your_test_split("data/gold/my_custom_tier/my_dataset")
raw_outputs = model.predict(test_inputs)

predictions = convert_model_outputs_to_delay_seconds(
    raw_outputs=raw_outputs,
    metadata=metadata,
    eval_table=eval_table,
)

metrics = evaluate_delay_predictions(
    eval_table=eval_table,
    predictions=predictions,
)

write_yaml(Path("runs/my_model/eval_metrics.yaml"), {"test": metrics})
```

The evaluator checks that the key columns exist and that the prediction
`future_delay_*` columns match the evaluation table. It also reports whether any
evaluation rows or prediction rows failed to align on the keys. For benchmark
comparison, predictions should cover all rows in the associated
`test_eval_table.parquet`; unmatched evaluation rows indicate missing
predictions.

## Add a Test-Evaluation Wrapper

For repeated benchmark runs, add a wrapper under `scripts/benchmark/test_eval/`.
The existing learning-based wrappers train and evaluate the model for each seed,
then aggregate the metrics into:

```text
test_eval_summary.yaml
```

The shared helper for repeated runs is:

```python
from src.benchmark.utils.test_eval import run_seed_sweep
```

Existing examples include:

```text
scripts/benchmark/test_eval/mlp.py
scripts/benchmark/test_eval/xgboost.py
scripts/benchmark/test_eval/lstm.py
scripts/benchmark/test_eval/transformer.py
scripts/benchmark/test_eval/gnn.py
```

Each wrapper builds a train command and an evaluation command for every seed.
The evaluation command should write `eval_metrics.yaml` in the seed run
directory. `run_seed_sweep` collects those files and writes the final summary.

## Run Your Evaluation

For a single evaluation script, use the dataset directory and the associated
core evaluation table:

```bash
python -m scripts.benchmark.eval.my_model \
  --data-dir data/gold/my_custom_tier/my_dataset \
  --test-eval-table data/gold/my_custom_tier/core/test_eval_table.parquet \
  --model-dir runs/benchmark/my_custom_tier/my_model
```

For a repeated test-evaluation wrapper:

```bash
python -m scripts.benchmark.test_eval.my_model \
  --data-dir data/gold/my_custom_tier/my_dataset \
  --test-eval-table data/gold/my_custom_tier/core/test_eval_table.parquet \
  --output-dir runs/benchmark/my_custom_tier/my_model \
  --config configs/benchmark/best_models/my_custom_tier/my_model.yaml \
  --seeds 0,1,2,3,4,5,6,7,8,9
```

The important invariant is that `--data-dir` and `--test-eval-table` refer to
the same Gold core that was used to create the evaluated dataset.

[Back to README](../README.md#what-do-you-want-to-do)
