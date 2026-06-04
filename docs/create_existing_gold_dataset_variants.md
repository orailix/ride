# Create Variants of Existing Model-Specific Gold Datasets

This tutorial describes how to run the existing Gold model-specific dataset
builders with alternate parameters.

Gold dataset construction does not use manifests. The Gold transforms are more
specific and diverse than the Bronze and Silver table transforms, so they are
implemented as dedicated command-line scripts under `scripts/dataset/gold/`.

If you changed the structure of the Silver tables, the existing Gold scripts may
no longer work without modification. The commands below assume that the Silver
outputs still provide the columns and table layout expected by the current Gold
builders.

The existing model-specific Gold builders can be run on any compatible Gold
core, either one provided by RIDE or a new core that you created yourself. These
scripts all consume the selected core through `--dataset-core-spec`, and their
own parameters can be changed independently of the core.

For example:

```bash
python -m scripts.dataset.gold.create_tabular_data \
  --dataset-core-spec data/gold/my_custom_tier/core/dataset_core_spec.yaml \
  --silver-dir data/silver \
  --nb-past-events 8 \
  --n-next-links 6 \
  --station-embedding-dim 12 \
  --missing-event-placeholder -999 \
  --output-dir data/gold/my_custom_tier/tabular
```

The same pattern applies to the sequential, GNN, and graph-event builders:

```bash
python -m scripts.dataset.gold.create_sequential_data \
  --dataset-core-spec data/gold/my_custom_tier/core/dataset_core_spec.yaml \
  --silver-dir data/silver \
  --nb-past-events 8 \
  --n-next-links 6 \
  --station-embedding-dim 12 \
  --missing-event-placeholder -999 \
  --output-dir data/gold/my_custom_tier/sequential

python -m scripts.dataset.gold.create_gnn_data \
  --dataset-core-spec data/gold/my_custom_tier/core/dataset_core_spec.yaml \
  --silver-dir data/silver \
  --nb-past-events 8 \
  --station-embedding-dim 12 \
  --missing-event-placeholder -999 \
  --graph-chunk-size 5000 \
  --output-dir data/gold/my_custom_tier/gnn

python -m scripts.dataset.gold.create_graph_event_data \
  --dataset-core-spec data/gold/my_custom_tier/core/dataset_core_spec.yaml \
  --silver-dir data/silver \
  --missing-event-placeholder -999 \
  --output-dir data/gold/my_custom_tier/graph_event
```

Change the model-specific parameters when the representation should change. For
example, `--nb-past-events` controls how much past train history is exported,
`--n-next-links` controls how much future path context is included by the
tabular and sequential builders, `--station-embedding-dim` controls the station
embedding size, and `--graph-chunk-size` controls the saved graph chunk size for
the GNN dataset.

[Back to README](../README.md#what-do-you-want-to-do)
