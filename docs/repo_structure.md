# Repository Structure

The repository separates reusable Python code, runnable scripts, run-level
configuration files, executable table manifests, documentation, notebooks, and
paper-artifact utilities. Commands in the tutorials are assumed to be run
from the repository root.

## Top-Level Layout

| Path | Purpose |
| --- | --- |
| `src/` | Reusable Python code for source downloads, dataset construction, benchmark models, and evaluation utilities. |
| `scripts/` | Command-line entry points for data download/build steps, model training/evaluation, hyperparameter search, and paper artifact generation. |
| `configs/` | Dataset pipeline settings, selected benchmark model configurations, and Optuna search spaces. |
| `manifests/` | Executable Bronze and Silver table specifications: source paths, output paths, transforms, parameters, checks, and field metadata. |
| `docs/` | Markdown setup, reference, extension, and reproducibility guides. |
| `notebooks/` | Notebook-based dataset and modeling walkthroughs. |

## Source Code

| Path | Purpose |
| --- | --- |
| `src/dataset/pipeline/` | Shared YAML, manifest-loading, manifest-execution, and output-writing helpers used by dataset pipelines. |
| `src/dataset/bronze/` | Helpers for downloading raw Infrabel source files used by the Bronze stage. |
| `src/dataset/silver/` | Silver relational dataset cleaning, enrichment, and table-construction utilities. |
| `src/dataset/gold/` | Gold benchmark-core construction, shared snapshot/evaluation-table helpers, and model-specific dataset builders. |
| `src/benchmark/models/` | Benchmark model and baseline implementations, plus model-specific prediction-table helpers. |
| `src/benchmark/utils/` | Shared evaluation, feature-ablation, precision, Optuna, and repeated-test-evaluation helpers. |

## Scripts

| Path | Purpose |
| --- | --- |
| `scripts/dataset/download/` | Download released RIDE datasets from Hugging Face, raw Infrabel source files, or raw Open-Meteo weather batches. |
| `scripts/dataset/bronze/` | Run the Bronze manifest-based pipeline on raw source files. |
| `scripts/dataset/silver/` | Run the Silver manifest-based pipeline and the separate weather concatenation step. |
| `scripts/dataset/gold/` | Build the Gold benchmark core and the tabular, sequential, GNN, and graph-event datasets. |
| `scripts/benchmark/train/` | Train benchmark models. |
| `scripts/benchmark/eval/` | Evaluate trained models and deterministic baselines against a test evaluation table. |
| `scripts/benchmark/test_eval/` | Run repeated train/evaluation procedures used for benchmark reporting. |
| `scripts/benchmark/optuna/` | Run hyperparameter optimization workflows. |
| `scripts/benchmark/ablation/` | Run MLP feature-family ablation workflows. |
| `scripts/benchmark/slurm/` | SLURM job templates for Lite and Standard Optuna jobs. |
| `scripts/paper/` | Table- and figure-generation scripts for paper reproducibility and documentation assets. |

## Configs and Manifests

| Path | Purpose |
| --- | --- |
| `configs/dataset/` | Dataset pipeline settings, including event months and Bronze/Silver manifest locations. |
| `configs/benchmark/best_models/` | Selected train/evaluation settings for Lite and Standard benchmark models. |
| `configs/benchmark/optuna/` | Hyperparameter search spaces for Lite and Standard benchmark tiers. |
| `manifests/bronze/` | Bronze table specifications describing raw source files, standardization rules, output paths, checks, and fields. |
| `manifests/silver/` | Silver table specifications describing upstream inputs, transform functions, parameters, output paths, checks, and fields. |

The manifest files are read by the dataset pipeline runner. They are therefore
both documentation and execution metadata: changing a manifest can change which
source files are read, which transform function is called, how columns are
renamed or normalized, which checks are enforced, and where outputs are written.

[Back to README](../README.md#what-do-you-want-to-do)
