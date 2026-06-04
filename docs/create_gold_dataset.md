# Create Your Own Model-Specific Gold Dataset

A model-specific Gold dataset is a representation of an existing Gold core for a
particular modeling setup. The Gold core defines the snapshot timestamps, the
active trains at those timestamps, and the benchmark targets. A model-specific
dataset builder should keep that benchmark definition fixed and only change how
the corresponding prediction instances are encoded.

## Add the Builder Files

Follow the same structure as the existing Gold builders:

- add reusable construction code under `src/dataset/gold/<name>_data.py`;
- add a command-line entry point under
  `scripts/dataset/gold/create_<name>_data.py`.

The command-line script should be thin. It should parse arguments and call the
builder function in `src/dataset/gold/<name>_data.py`. Existing examples include
`scripts/dataset/gold/create_tabular_data.py`,
`scripts/dataset/gold/create_sequential_data.py`,
`scripts/dataset/gold/create_gnn_data.py`, and
`scripts/dataset/gold/create_graph_event_data.py`.

At minimum, the new command should accept:

- `--dataset-core-spec`, pointing to the Gold core specification;
- `--silver-dir`, pointing to the Silver dataset root;
- `--output-dir`, where the new model-specific dataset will be written;
- any parameters specific to the new representation.

## Read the Core and Silver Data

The builder should start by reading the Gold core specification:

```python
from src.dataset.pipeline.helpers import read_yaml

cfg = read_yaml(dataset_core_spec)
```

The core specification contains the `train_snapshots`, `test_snapshots`, and
core parameters such as `n_future`, `idle_time_beg`, and `idle_time_end`.

Gold builders usually read Silver inputs from `--silver-dir`:

```python
events_dir = silver_dir / "events"
journeys_dir = silver_dir / "journeys"
weather_path = silver_dir / "static" / "weather.parquet"
node_links_path = silver_dir / "static" / "node_links.parquet"
op_nodes_path = silver_dir / "static" / "op_nodes.parquet"
```

Only load the static tables that are needed by your representation.

## Build the Snapshot Index

Most model-specific builders should use the shared Gold indexing helper:

```python
from src.dataset.gold.helpers import build_indexes

index = build_indexes(
    events_dir=events_dir,
    journeys_dir=journeys_dir,
    snapshot_config=cfg,
    splits_to_build=["train", "test"],
    index_events_optional_get=["event_arr_line", "event_dep_line"],
    index_journeys_optional_get=["train_relation", "operator"],
    show_progress=True,
)
```

The index contains compact arrays for the Silver events and journey activity
windows that overlap the selected core snapshots. If your dataset needs extra
fields, request them through the optional arguments supported by `build_indexes`,
such as `index_events_optional_get` or `index_journeys_optional_get`. Depending
on the selected core and optional fields, this step can still require
substantial RAM because the relevant event and journey arrays are held in
memory.

## Iterate Over Snapshots

Each snapshot represents a prediction time. For each snapshot, use the journey
activity windows from the index to identify the trains that are active at that
timestamp:

```python
import pandas as pd

from src.dataset.gold.helpers import get_active_mask_for_snapshot

train_snapshots = pd.DatetimeIndex(pd.to_datetime(cfg["train_snapshots"]))
test_snapshots = pd.DatetimeIndex(pd.to_datetime(cfg["test_snapshots"]))

journeys_index = index["journeys"]
events_index = index["events"]

for split_name, snapshots in {
    "train": train_snapshots,
    "test": test_snapshots,
}.items():
    for snapshot_ts in snapshots:
        active_mask = get_active_mask_for_snapshot(
            journeys_index["appearance_start"],
            journeys_index["disappearance_end"],
            snapshot_ts,
        )
        active_indices = active_mask.nonzero()[0]

        for idx in active_indices:
            # Gather the information needed for this active journey and encode
            # it in the format required by your model-specific dataset.
            # For example:
            train_journey = {
                name: values[idx]
                for name, values in journeys_index.items()
            }
            train_id = train_journey["train_ids"]
            service_date = train_journey["service_dates"]
            start, end = events_index["key_slices"][(train_id, service_date)]
            train_events = {
                name: values[start:end]
                for name, values in events_index.items()
                if name != "key_slices"
            }
```

The active trains returned by this mask define the prediction instances for that
snapshot. A model-specific builder should create one encoded example for each
active train unless the representation intentionally groups several active
trains into a single graph or sequence object.

Inside this loop, you can construct whatever representation your model needs.
For example, an ML dataset might gather past events for the active journey and
turn them into feature columns or tensors. The existing builders in
`src/dataset/gold/tabular_data.py`, `src/dataset/gold/sequential_data.py`,
`src/dataset/gold/gnn_data.py`, and `src/dataset/gold/graph_event_data.py`
provide concrete examples of different representation choices.

When building features inside this loop, be careful about leakage. The input for
a prediction instance may use information available at `snapshot_ts`, but it
must not include realized outcomes from after the snapshot time. Do not include
future delays, future observed timestamps, weather observations from after the
snapshot time, or features derived from event outcomes after the snapshot time.
Future schedule information may be valid when it is known at the snapshot time,
such as planned timestamps, future operational points, event types, or planned
path context. Static infrastructure, station metadata, past events, current or
last-known delay, and planned schedule information known at the snapshot time
are valid inputs.

When a representation depends on a fixed number of past or future scheduled
events, each train journey may need to be converted into fixed-width arrays
around the snapshot time. The helper `get_padded_arrays_from_components` in
`src/dataset/gold/helpers.py` pads each journey before and after its observed
events, so builders can select consistent past and future slots even near the
beginning or end of a journey.

## Write the Dataset Outputs

Write outputs under `--output-dir` and keep the train/test split explicit. The
exact file format depends on the representation. Existing builders write numpy
arrays, parquet tables, or serialized graph-event artifacts.

Also write enough metadata for downstream training and evaluation code to
understand the representation and align predictions back to the Gold core. Each
encoded example should preserve `ts`, `train_id`, and `service_date`, or an
equivalent set of keys that can be joined back to the core evaluation instances.
Depending on the dataset, this may include:

- a schema file such as `scheme.yaml` or `feature_spec.yaml`;
- normalization statistics such as `normalization.yaml`;
- metadata describing builder parameters and input paths.

If the dataset uses normalization, fit normalization statistics on the training
split only, then apply the same statistics to both train and test outputs.

[Back to README](../README.md#what-do-you-want-to-do)
