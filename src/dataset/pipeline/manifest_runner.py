"""Manifest execution utilities for dataset pipeline layers."""

import importlib
import inspect
import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def _apply_rename(df: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
    """Rename columns according to a manifest rename map."""
    return df.rename(columns=rename_map) if rename_map else df


def _apply_drop(df: pd.DataFrame, drop_list) -> pd.DataFrame:
    """Drop manifest-selected columns, including simple wildcard patterns."""
    if not drop_list:
        return df
    cols = list(df.columns)
    to_drop = set()
    for pat in drop_list:
        if "*" in pat:
            regex = "^" + re.escape(pat).replace("\\*", ".*") + "$"
            to_drop.update([c for c in cols if re.match(regex, c)])
        elif pat in cols:
            to_drop.add(pat)
    if to_drop:
        df = df.drop(columns=[c for c in to_drop if c in df.columns])
    return df


def _coerce_dtype(series: pd.Series, to_type: str) -> pd.Series:
    """Coerce a pandas Series to one of the manifest-supported logical types."""
    t = to_type.lower()
    if t in ("str", "string"):
        return series.astype("string").astype(str)
    if t in ("float", "double"):
        return pd.to_numeric(series, errors="coerce").astype("float64")
    if t in ("int", "integer"):
        return pd.to_numeric(series, errors="coerce").astype("Int64")
    if t in ("datetime", "timestamp"):
        return pd.to_datetime(series, errors="coerce", utc=False)
    return series


def _apply_normalize(df: pd.DataFrame, normalize_cfg: dict) -> pd.DataFrame:
    """Apply manifest normalization rules to dataframe columns."""
    if not normalize_cfg:
        return df
    for col, rules in normalize_cfg.items():
        if "split" in rules and col in df.columns:
            delim = rules["split"].get("delimiter", ";")
            targets = rules["split"]["into"]
            to_type = rules["split"].get("to_type", "str")
            parts = df[col].astype(str).str.split(delim, n=len(targets) - 1, expand=True)
            parts.columns = targets
            for target_col in targets:
                if target_col in parts.columns:
                    parts[target_col] = _coerce_dtype(parts[target_col], to_type)
            df = pd.concat([df.drop(columns=[col]), parts], axis=1)
            continue
        if col not in df.columns:
            continue
        series = df[col]
        to_type = rules.get("to_type")
        if rules.get("strip"):
            series = series.astype(str).str.strip()
        if rules.get("upper"):
            series = series.astype(str).str.upper()
        if to_type:
            series = _coerce_dtype(series, to_type)
        df[col] = series
    return df


def _run_checks(df: pd.DataFrame, checks: list):
    """Run manifest validation checks on the transformed dataframe."""
    if not checks:
        return
    for check in checks:
        if "assert_unique" in check:
            cols = check["assert_unique"]
            dup = df.duplicated(subset=cols, keep=False)
            if dup.any():
                sample = df.loc[dup, cols].head(5).to_dict(orient="records")
                raise AssertionError(f"assert_unique failed on {cols}; examples: {sample}")
        if "assert_not_null" in check:
            cols = check["assert_not_null"]
            for col in cols:
                if col in df.columns and df[col].isna().any():
                    raise AssertionError(f"assert_not_null failed on column '{col}'")
        if "assert_in_range" in check:
            cfg = check["assert_in_range"]
            col = cfg["column"]
            mn, mx = cfg["min"], cfg["max"]
            if col in df.columns:
                bad = df[(df[col].notna()) & ((df[col] < mn) | (df[col] > mx))]
                if not bad.empty:
                    print(
                        f"warning: {len(bad)} rows out of range for '{col}' "
                        f"[{mn},{mx}]"
                    )


def _load_yaml_text(path: str):
    """Decode a text file by trying the encodings used by source manifests."""
    data = Path(path).read_bytes()
    for enc in ("utf-8", "utf-8-sig", "utf-16", "utf-16le", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"Cannot decode {path}")


def _load_source_dataframe(path: str, fmt: str, spec: dict) -> pd.DataFrame:
    """Load one source dataframe from a file, glob, or directory."""
    path_obj = Path(path)
    sep = spec.get("sep")
    encoding = spec.get("encoding", "utf-8")

    def _read_single(single_path: Path) -> pd.DataFrame:
        """Read one source file in the requested manifest format."""
        if fmt == "parquet":
            return pd.read_parquet(single_path)
        if fmt == "csv":
            return pd.read_csv(
                single_path,
                sep=sep if sep is not None else ",",
                encoding=encoding,
            )
        raise ValueError(f"Unsupported format '{fmt}' for source {single_path}")

    if "*" in path_obj.name:
        frames = [_read_single(p) for p in sorted(path_obj.parent.glob(path_obj.name))]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if path_obj.is_dir():
        frames = [_read_single(p) for p in sorted(path_obj.glob("*")) if p.is_file()]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return _read_single(path_obj)


def _call_transform_function(
    transform_cfg: dict,
    sources: Dict[str, object],
    wildcards: Optional[Dict[str, str]],
) -> pd.DataFrame:
    """Import and call the transform function declared by a manifest."""
    function_path = transform_cfg.get("function")
    if not function_path:
        raise ValueError("Transform function path must be provided.")
    module_name, func_name = function_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    transform_fn = getattr(module, func_name)

    sig = inspect.signature(transform_fn)
    kwargs: Dict[str, object] = {}
    if "sources" in sig.parameters:
        kwargs["sources"] = {
            key: value.copy() if isinstance(value, pd.DataFrame) else value
            for key, value in sources.items()
        }
    if "wildcards" in sig.parameters:
        kwargs["wildcards"] = wildcards or {}
    if "params" in sig.parameters:
        kwargs["params"] = transform_cfg.get("params") or {}

    return transform_fn(**kwargs)


def apply_manifest(manifest: dict, wildcards: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Load sources, apply transform steps, write output, and return the dataframe."""
    io_cfg = manifest["io"]
    transform_cfg = manifest.get("transform", {})

    sources_cfg = io_cfg.get("sources") or {}
    if not sources_cfg:
        raise ValueError("Manifest must define io.sources.")

    sources: Dict[str, object] = {}
    for name, spec in sources_cfg.items():
        raw_path = spec.get("path")
        if not raw_path:
            raise ValueError(f"Source '{name}' is missing a path.")
        path = raw_path
        if wildcards:
            token = wildcards.get(name) or wildcards.get("main")
            if token:
                path = path.replace("*", token)
        load_mode = str(spec.get("load", "eager")).lower()
        fmt = str(spec.get("format", "csv")).lower()
        if load_mode == "lazy":
            sources[name] = path
            continue
        df_source = _load_source_dataframe(path, fmt, spec)
        sources[name] = df_source

    has_transform_fn = bool(transform_cfg.get("function"))

    if not has_transform_fn:
        if len(sources) != 1:
            raise ValueError("Simple manifests require exactly one source.")
        main_value = next(iter(sources.values()))
        if not isinstance(main_value, pd.DataFrame):
            raise ValueError("Single source must be eagerly loaded.")
        df = main_value.copy()
    else:
        df = _call_transform_function(transform_cfg, sources, wildcards)
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Transform function must return a pandas DataFrame.")

    df = _apply_rename(df, transform_cfg.get("rename", {}))
    df = _apply_drop(df, transform_cfg.get("drop", []))
    df = _apply_normalize(df, transform_cfg.get("normalize", {}))

    _run_checks(df, manifest.get("checks", []))

    output_cfg = io_cfg.get("output")
    if not output_cfg:
        raise ValueError("Manifest must define io.output.")
    out_path = output_cfg.get("path")
    if not out_path:
        raise ValueError("io.output must include a path.")
    if wildcards:
        token = wildcards.get("output") or wildcards.get("main")
        if token:
            out_path = out_path.replace("*", token)
    out_format = str(output_cfg.get("format", "parquet")).lower()

    out_path_obj = Path(out_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    if out_format == "parquet":
        df.to_parquet(out_path_obj, index=False)
    elif out_format == "csv":
        df.to_csv(out_path_obj, index=False)
    else:
        raise ValueError(f"Unsupported output format '{out_format}'.")

    print(f"Built {manifest['id']} -> {out_path_obj} ({len(df):,} rows)")
    return df

