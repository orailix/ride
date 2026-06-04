# Rebuild the Data Pipeline

This page gives the commands used to rebuild the RIDE data pipeline from
source downloads: Raw, Bronze, Silver, and the Gold Lite/Standard benchmark
datasets.

For exact paper reproduction, prefer the released Silver and Gold datasets
linked from the repository README. Rebuilding from source downloads is useful
for auditing or extending the pipeline, but it is not guaranteed to reproduce
the released downstream datasets byte-for-byte because the hosted source files
are controlled by external providers. This is especially true for Infrabel
infrastructure files that can change as the network evolves:

- `operationele-punten-van-het-netwerk.csv`: operational points;
- `lijnsecties.csv`: railway line sections.

Monthly punctuality files and Open-Meteo archive responses are also downloaded
from external services, so availability, schemas, corrections, or API behavior
may change over time.

## Configuration

The paper pipeline uses:

```bash
configs/dataset/main_dataset.yaml
```

This configuration points to the Bronze and Silver manifests and lists the
event months used by the release, from `202301` to `202512`.

## Raw and Bronze

Download the raw Infrabel source files:

```bash
python -m scripts.dataset.download.download_bronze_sources \
  configs/dataset/main_dataset.yaml
```

The raw source files are stored under `data/raw/`:

- `data/raw/static/operationele-punten-van-het-netwerk.csv`;
- `data/raw/static/lijnsecties.csv`;
- `data/raw/events/Data_raw_punctuality_<YYYYMM>.csv`;
- `data/raw/metadata.yaml`.

Build the Bronze tables:

```bash
python -m scripts.dataset.bronze.run_bronze_pipeline \
  configs/dataset/main_dataset.yaml
```

This creates standardized Bronze tables under `data/bronze/`.

## Silver

Build the non-weather Silver tables:

```bash
python -m scripts.dataset.silver.run_silver_pipeline \
  configs/dataset/main_dataset.yaml
```

This creates `events/`, `journeys/`, and the static railway tables under
`data/silver/`.

Download Open-Meteo weather batches after Silver operational points have been
built. The released Silver operational-point table contains 1,355 operational
points, so with `--chunk-size 100` this requires 14 chunks: one command for each
`--chunk-start` value from `0` to `1300` in steps of `100`. The observed
Open-Meteo free-tier limits for these paper-scale chunk requests were
approximately 1 request per minute, 2 requests per hour, and 3 requests per day.
Run these downloads manually or with a scheduler at a cadence that respects the
current API limits:

```bash
python -m scripts.dataset.download.download_weather_data \
  configs/dataset/main_dataset.yaml \
  --chunk-start 0 \
  --chunk-size 100

python -m scripts.dataset.download.download_weather_data \
  configs/dataset/main_dataset.yaml \
  --chunk-start 100 \
  --chunk-size 100

# Continue with --chunk-start 200, 300, ..., 1300.
```

The raw weather batches are stored under `data/raw/weather/` as parquet files,
and their download metadata is appended to `data/raw/metadata.yaml`.

Concatenate the raw weather batches into the Silver weather table:

```bash
python -m scripts.dataset.silver.run_silver_weather
```

The complete Silver release is then under `data/silver/`.

## Gold Core

Build Gold Lite core:

```bash
python -m scripts.dataset.gold.create_gold_dataset_core \
  --start-train-day 2023-01-01 \
  --end-train-day 2024-12-31 \
  --start-test-day 2025-01-01 \
  --end-test-day 2025-12-31 \
  --n-train 15000 \
  --n-test 3000 \
  --n-future 15 \
  --idle-time-beg 5 \
  --idle-time-end 5 \
  --seed 42 \
  --events-dir data/silver/events \
  --journeys-dir data/silver/journeys \
  --missing-event-placeholder -1 \
  --output-root data/gold/lite/core
```

Build Gold Standard core:

```bash
python -m scripts.dataset.gold.create_gold_dataset_core \
  --start-train-day 2023-01-01 \
  --end-train-day 2024-12-31 \
  --start-test-day 2025-01-01 \
  --end-test-day 2025-12-31 \
  --n-train 50000 \
  --n-test 10000 \
  --n-future 15 \
  --idle-time-beg 5 \
  --idle-time-end 5 \
  --seed 42 \
  --events-dir data/silver/events \
  --journeys-dir data/silver/journeys \
  --missing-event-placeholder -1 \
  --output-root data/gold/standard/core
```

## Gold Model-Specific Datasets

Build the four Gold Lite model-specific datasets:

```bash
python -m scripts.dataset.gold.create_tabular_data \
  --dataset-core-spec data/gold/lite/core/dataset_core_spec.yaml \
  --nb-past-events 15 \
  --n-next-links 10 \
  --station-embedding-dim 8 \
  --missing-event-placeholder -1 \
  --output-dir data/gold/lite/tabular

python -m scripts.dataset.gold.create_sequential_data \
  --dataset-core-spec data/gold/lite/core/dataset_core_spec.yaml \
  --nb-past-events 15 \
  --n-next-links 10 \
  --station-embedding-dim 8 \
  --missing-event-placeholder -1 \
  --output-dir data/gold/lite/sequential

python -m scripts.dataset.gold.create_gnn_data \
  --dataset-core-spec data/gold/lite/core/dataset_core_spec.yaml \
  --nb-past-events 15 \
  --station-embedding-dim 8 \
  --missing-event-placeholder -1 \
  --graph-chunk-size 10000 \
  --output-dir data/gold/lite/gnn

python -m scripts.dataset.gold.create_graph_event_data \
  --dataset-core-spec data/gold/lite/core/dataset_core_spec.yaml \
  --missing-event-placeholder -1 \
  --output-dir data/gold/lite/graph_event
```

Build the four Gold Standard model-specific datasets:

```bash
python -m scripts.dataset.gold.create_tabular_data \
  --dataset-core-spec data/gold/standard/core/dataset_core_spec.yaml \
  --nb-past-events 15 \
  --n-next-links 10 \
  --station-embedding-dim 8 \
  --missing-event-placeholder -1 \
  --output-dir data/gold/standard/tabular

python -m scripts.dataset.gold.create_sequential_data \
  --dataset-core-spec data/gold/standard/core/dataset_core_spec.yaml \
  --nb-past-events 15 \
  --n-next-links 10 \
  --station-embedding-dim 8 \
  --missing-event-placeholder -1 \
  --output-dir data/gold/standard/sequential

python -m scripts.dataset.gold.create_gnn_data \
  --dataset-core-spec data/gold/standard/core/dataset_core_spec.yaml \
  --nb-past-events 15 \
  --station-embedding-dim 8 \
  --missing-event-placeholder -1 \
  --graph-chunk-size 10000 \
  --output-dir data/gold/standard/gnn

python -m scripts.dataset.gold.create_graph_event_data \
  --dataset-core-spec data/gold/standard/core/dataset_core_spec.yaml \
  --missing-event-placeholder -1 \
  --output-dir data/gold/standard/graph_event
```

[Back to README](../README.md#what-do-you-want-to-do)
