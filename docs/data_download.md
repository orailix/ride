# Downloading the Released Datasets

Released RIDE data assets are hosted on Hugging Face. The repository provides a
small downloader that places each asset under the local `data/` directory used
by the scripts and notebooks.

Run commands from the repository root.

## Download a Full Release

Download the Silver relational release:

```bash
python -m scripts.dataset.download.download_dataset --target silver
```

This writes to:

```text
data/silver/
```

Download the Gold Lite benchmark release:

```bash
python -m scripts.dataset.download.download_dataset --target gold_lite
```

This writes to:

```text
data/gold/lite/
```

Download the Gold Standard benchmark release:

```bash
python -m scripts.dataset.download.download_dataset --target gold_standard
```

This writes to:

```text
data/gold/standard/
```

## Download Only One Gold Component

Gold releases contain a shared `core` plus model-specific datasets. You can
download only the component you need:

```bash
python -m scripts.dataset.download.download_dataset --target gold_lite_core
python -m scripts.dataset.download.download_dataset --target gold_lite_tabular
python -m scripts.dataset.download.download_dataset --target gold_lite_sequential
python -m scripts.dataset.download.download_dataset --target gold_lite_gnn
python -m scripts.dataset.download.download_dataset --target gold_lite_graph_event
```

The same component targets are available for the Standard tier:

```bash
python -m scripts.dataset.download.download_dataset --target gold_standard_core
python -m scripts.dataset.download.download_dataset --target gold_standard_tabular
python -m scripts.dataset.download.download_dataset --target gold_standard_sequential
python -m scripts.dataset.download.download_dataset --target gold_standard_gnn
python -m scripts.dataset.download.download_dataset --target gold_standard_graph_event
```

## Choose a Different Output Directory

Each target has a default local path, but you can override it:

```bash
python -m scripts.dataset.download.download_dataset \
  --target gold_lite_tabular \
  --output-dir /path/to/ride-gold-lite-tabular
```

[Back to README](../README.md#what-do-you-want-to-do)
