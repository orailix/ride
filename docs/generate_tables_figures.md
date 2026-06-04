# Generate Tables and Figures

The scripts under `scripts/paper/` regenerate paper tables and figures from
released datasets and benchmark run outputs.

Generated tables and figures are written under:

```text
tables_figures/
```

Within that folder, paper tables are written to `tables_figures/tables/` and
paper figures are written to `tables_figures/figures/`.

The result scripts read benchmark output summaries produced by the reproduction
commands. The dataset figure scripts read released dataset files.

## Standard Results

Generate the Gold Standard result tables and figures with:

```bash
python -m scripts.paper.generate_standard_tables_figures
```

## Lite Results

Generate the Gold Lite result tables and figures with:

```bash
python -m scripts.paper.generate_lite_tables_figures
```

## Ablation Results

Generate the feature-ablation result table and figure with:

```bash
python -m scripts.paper.generate_ablation_tables_figures
```

## Dataset Figures

Generate the paper data figures with:

```bash
python -m scripts.paper.generate_dataset_figures
```

[Back to README](../README.md#what-do-you-want-to-do)
