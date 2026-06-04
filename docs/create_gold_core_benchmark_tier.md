# Create a New Gold-Core Benchmark Tier

This tutorial describes how to create a new Gold-core benchmark tier. A Gold
core defines the benchmark prediction instances, train/test split, future
prediction horizon, and evaluation reference tables.

If you changed the structure of the Silver event or journey tables, the existing
Gold-core builder may no longer work without modification. The command below
assumes that the Silver outputs still provide the columns and table layout
expected by the current core builder.

The command writes `dataset_core_spec.yaml` under `--output-root`;
model-specific Gold builders use that file as their input. This allows you to
create a new Gold tier, analogous to the provided Lite and Standard tiers, with
your own split dates, snapshot counts, prediction horizon, and idle-time
settings.

The most important arguments are:

- `--start-train-day` and `--end-train-day`, which define the inclusive training
  period;
- `--start-test-day` and `--end-test-day`, which define the inclusive test
  period;
- `--n-train` and `--n-test`, which define how many snapshot timestamps are
  sampled for each split;
- `--n-future`, which defines how many future events are predicted for each
  active train;
- `--idle-time-beg` and `--idle-time-end`, which define how long trains remain
  active before the beginning and after the end of their observed journey;
- `--seed`, which controls snapshot sampling;
- `--events-dir` and `--journeys-dir`, which point to the Silver event and
  journey tables used to build the core;
- `--output-root`, which defines the directory for the new core tier.

For example:

```bash
python -m scripts.dataset.gold.create_gold_dataset_core \
  --start-train-day 2024-01-01 \
  --end-train-day 2024-06-30 \
  --start-test-day 2024-07-01 \
  --end-test-day 2024-09-30 \
  --n-train 2500 \
  --n-test 500 \
  --n-future 10 \
  --idle-time-beg 10 \
  --idle-time-end 15 \
  --seed 123 \
  --events-dir data/silver/events \
  --journeys-dir data/silver/journeys \
  --missing-event-placeholder -999 \
  --output-root data/gold/my_custom_tier/core
```

[Back to README](../README.md#what-do-you-want-to-do)
