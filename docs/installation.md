# Installation

RIDE requires Python 3.11.

All commands in the tutorials assume they are run from the repository root.

## Create an Environment

```bash
conda create -n ride python=3.11
conda activate ride
```

## Install Dependencies

The requirements file is grouped by usage. The first block contains the common
dependencies, followed by optional blocks for dataset construction, benchmark
models and hyperparameter search, and plotting or notebook workflows. If you
only need part of the repository, you can comment out unused blocks before
installing.

PyTorch installation can depend on the local CUDA setup. If the default
installation does not match your hardware, follow the official PyTorch
installation selector and then rerun the command above for the remaining
dependencies.

From the repository root:

```bash
pip install -r requirements.txt
```

## Check the Installation

```bash
python -c "import pandas, pyarrow, torch, torch_geometric; print('RIDE environment ready')"
```

[Back to README](../README.md#what-do-you-want-to-do)
