"""Download published RIDE dataset releases from Hugging Face."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import snapshot_download


TARGET_SPECS = {
    "silver": {
        "repo_id": "orailix/ride-silver",
        "default_output_dir": Path("data/silver"),
        "subdir": None,
    },
    "gold_standard": {
        "repo_id": "orailix/ride-gold-standard",
        "default_output_dir": Path("data/gold/standard"),
        "subdir": None,
    },
    "gold_lite": {
        "repo_id": "orailix/ride-gold-lite",
        "default_output_dir": Path("data/gold/lite"),
        "subdir": None,
    },
    "gold_standard_core": {
        "repo_id": "orailix/ride-gold-standard",
        "default_output_dir": Path("data/gold/standard/core"),
        "subdir": "core",
    },
    "gold_standard_tabular": {
        "repo_id": "orailix/ride-gold-standard",
        "default_output_dir": Path("data/gold/standard/tabular"),
        "subdir": "tabular",
    },
    "gold_standard_sequential": {
        "repo_id": "orailix/ride-gold-standard",
        "default_output_dir": Path("data/gold/standard/sequential"),
        "subdir": "sequential",
    },
    "gold_standard_gnn": {
        "repo_id": "orailix/ride-gold-standard",
        "default_output_dir": Path("data/gold/standard/gnn"),
        "subdir": "gnn",
    },
    "gold_standard_graph_event": {
        "repo_id": "orailix/ride-gold-standard",
        "default_output_dir": Path("data/gold/standard/graph_event"),
        "subdir": "graph_event",
    },
    "gold_lite_core": {
        "repo_id": "orailix/ride-gold-lite",
        "default_output_dir": Path("data/gold/lite/core"),
        "subdir": "core",
    },
    "gold_lite_tabular": {
        "repo_id": "orailix/ride-gold-lite",
        "default_output_dir": Path("data/gold/lite/tabular"),
        "subdir": "tabular",
    },
    "gold_lite_sequential": {
        "repo_id": "orailix/ride-gold-lite",
        "default_output_dir": Path("data/gold/lite/sequential"),
        "subdir": "sequential",
    },
    "gold_lite_gnn": {
        "repo_id": "orailix/ride-gold-lite",
        "default_output_dir": Path("data/gold/lite/gnn"),
        "subdir": "gnn",
    },
    "gold_lite_graph_event": {
        "repo_id": "orailix/ride-gold-lite",
        "default_output_dir": Path("data/gold/lite/graph_event"),
        "subdir": "graph_event",
    },
}


def copy_directory_contents(source_dir: Path, output_dir: Path) -> None:
    """Copy all files and directories from a downloaded subdirectory to the target path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in source_dir.iterdir():
        destination = output_dir / child.name
        if child.is_dir():
            shutil.copytree(child, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(child, destination)


def download_target(target: str, output_dir: Path) -> None:
    """Download one configured dataset target to the requested local output directory."""
    spec = TARGET_SPECS[target]
    repo_id = spec["repo_id"]
    subdir = spec["subdir"]
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    if subdir is None:
        snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=output_dir)
        return

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=tmp_path,
            allow_patterns=[f"{subdir}/**"],
        )
        source_dir = tmp_path / subdir
        if not source_dir.exists():
            raise FileNotFoundError(f"Subdirectory '{subdir}' was not found in dataset repo '{repo_id}'.")
        copy_directory_contents(source_dir, output_dir)


def main() -> None:
    """Parse CLI arguments and download the selected published dataset artifact."""
    parser = argparse.ArgumentParser(description="Download a published dataset artifact from Hugging Face.")
    parser.add_argument("--target", required=True, choices=TARGET_SPECS, help="Dataset artifact to download.")
    parser.add_argument("--output-dir", type=Path, help="Destination directory. Defaults to the standard local dataset path for the selected target.")
    args = parser.parse_args()

    output_dir = args.output_dir or TARGET_SPECS[args.target]["default_output_dir"]
    print(f"[dataset-download] downloading target='{args.target}' to '{output_dir}'")
    download_target(args.target, output_dir)
    print(f"[dataset-download] download complete for target='{args.target}'")


if __name__ == "__main__":
    main()
