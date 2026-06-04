# Modify the Download, Bronze, and Silver Pipeline

This tutorial describes how to modify the RIDE data pipeline across its
download, Bronze, and Silver layers.

## Create a Configuration File

Pipeline commands take a dataset configuration file as input. The provided paper
configuration is:

```bash
configs/dataset/main_dataset.yaml
```

For experiments or extensions, create your own configuration file and pass it to
the pipeline commands:

```bash
path/to/your/config.yaml
```

For example:

```yaml
bronze_manifest_dir: path/to/your/bronze_manifests
silver_manifest_dir: path/to/your/silver_manifests
events:
  months:
    - "202401"
    - "202402"
    - "202403"
```

This allows you to change the event months or point the pipeline to alternate
Bronze and Silver manifest directories without modifying the provided paper
configuration.

## Modify the Download

After creating your own configuration file, download the corresponding raw
source files. This command downloads the raw Infrabel sources used by the
pipeline, including the static infrastructure files and the selected event
months:

```bash
python -m scripts.dataset.download.download_bronze_sources \
  path/to/your/config.yaml
```

## Modify Bronze

Bronze tables are defined by the manifests in the directory referenced by
`bronze_manifest_dir` in your configuration file. The provided manifests live in
`manifests/bronze/`. For custom Bronze changes, copy the provided manifests into
your own manifest directory and point `bronze_manifest_dir` to that directory.
These manifests describe how raw source files are converted into standardized
parquet tables under `data/bronze/`.

For most Bronze changes, edit the relevant manifest:

- `manifests/bronze/events.yaml` for monthly punctuality event files;
- `manifests/bronze/op_nodes.yaml` for operational points;
- `manifests/bronze/line_sections.yaml` for railway line sections.

The main sections to modify are:

- `io.sources`, to change the raw input path, file format, separator, or
  encoding;
- `io.output`, to change the Bronze output location or format;
- `transform.rename`, to map source column names to RIDE column names;
- `transform.drop`, to remove raw columns that should not be kept in Bronze;
- `transform.normalize`, to strip strings, change case, split columns, or coerce
  column types;
- `checks`, to enforce simple validation rules such as uniqueness or non-null
  columns;
- `fields` and `notes`, to keep the manifest documentation aligned with the
  table schema.

To run the Bronze pipeline with your configuration:

```bash
python -m scripts.dataset.bronze.run_bronze_pipeline \
  path/to/your/config.yaml
```

## Modify Silver

Silver tables are defined by the manifests in the directory referenced by
`silver_manifest_dir` in your configuration file. The provided manifests live in
`manifests/silver/`.

If you changed the structure of the Bronze tables, the existing Silver
transforms may no longer work without modification. The guidance below assumes
that the Bronze outputs still provide the columns and table layout expected by
the current Silver manifests and transform functions.

For many changes, you can copy the provided Silver manifests into your own
manifest directory, keep the existing Silver flow, and only edit the parameters
passed to the transform functions. These parameters are defined under the
`transform.params` section of the relevant Silver manifest. For example:

- in `op_nodes.yaml`, update `manual_delete_op_ids` to remove operational points
  from the Silver node table;
- in `op_nodes.yaml`, update `manual_overrides` to add or correct operational
  point coordinates;
- in `line_sections.yaml`, update `matching_epsilon_m` to change the tolerance
  used when matching operational points to line-section geometries;
- in `line_sections.yaml`, update `manual_delete_line_section_ids` to remove
  line sections from the Silver infrastructure table;
- in `events.yaml`, update delay and monotonicity thresholds such as
  `max_late_sec`, `max_early_sec`, or `observed_monotonic_threshold_sec`.

If your change requires different Silver construction logic, edit the
`transform.function` entry in your Silver manifest to reference a custom Python
transform function. The function should return a pandas DataFrame and can
receive manifest sources, wildcards, and parameters through the standard
manifest runner interface.

To run the Silver pipeline with your configuration:

```bash
python -m scripts.dataset.silver.run_silver_pipeline \
  path/to/your/config.yaml
```

After building a custom Silver dataset, download the raw weather files for all
chunks of the new Silver operational-node table, then build the Silver weather
table:

```bash
python -m scripts.dataset.download.download_weather_data \
  path/to/your/config.yaml \
  --nodes-path path/to/your/silver/static/op_nodes.parquet \
  --chunk-start 0 \
  --chunk-size 100

# Repeat with --chunk-start 100, 200, ... until all operational points
# in the custom Silver op_nodes table have been covered.

python -m scripts.dataset.silver.run_silver_weather \
  --manifest path/to/your/silver_manifests/weather.yaml
```

The weather step is separate because the raw Open-Meteo requests depend on the
Silver operational points. The provided weather manifest lives at
`manifests/silver/weather.yaml`; for custom weather processing, copy and modify
that manifest as part of your custom Silver manifest directory. If your custom
Silver operational-node table is not written to the default
`data/silver/static/op_nodes.parquet`, pass its path with `--nodes-path` when
downloading weather.

[Back to README](../README.md#what-do-you-want-to-do)
