#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute a lightweight summary for AQVolt26 parquet data.")
    parser.add_argument(
        "parquet_path",
        nargs="?",
        default="data/aqvolt26/test.parquet",
        help="Path to AQVolt26 parquet file.",
    )
    parser.add_argument(
        "--output-json",
        default="data/aqvolt26/analysis_summary.json",
        help="Where to write the machine-readable summary.",
    )
    return parser.parse_args()


def safe_numeric_stats(series: pd.Series) -> dict[str, float | int | None]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    values = clean.tolist()
    return {
        "count": len(values),
        "min": float(min(values)),
        "median": float(statistics.median(values)),
        "mean": float(statistics.fmean(values)),
        "max": float(max(values)),
    }


def top_counter(counter: Counter[Any], n: int = 10) -> list[dict[str, Any]]:
    return [{"key": key, "count": count} for key, count in counter.most_common(n)]


def normalize_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def main() -> int:
    args = parse_args()
    parquet_path = Path(args.parquet_path)
    if not parquet_path.exists():
        print(f"Missing parquet file: {parquet_path}", file=sys.stderr)
        return 2

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as exc:
        print(
            "Failed to read parquet. Install a parquet engine such as pyarrow, then retry.\n"
            f"Underlying error: {exc}",
            file=sys.stderr,
        )
        return 1

    summary: dict[str, Any] = {
        "parquet_path": str(parquet_path.resolve()),
        "num_rows": int(len(df)),
        "num_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "numeric_stats": {},
    }

    for col in ["nsites", "fmax", "coh_epa", "energy", "frame_id", "temperature"]:
        if col in df.columns:
            summary["numeric_stats"][col] = safe_numeric_stats(df[col])

    if "elements" in df.columns:
        element_counter: Counter[str] = Counter()
        for row in df["elements"].dropna():
            if isinstance(row, (list, tuple)):
                element_counter.update(str(item) for item in row)
        summary["top_elements"] = top_counter(element_counter)

    if "composition" in df.columns:
        formula_counter: Counter[str] = Counter()
        for row in df["composition"].dropna():
            if isinstance(row, dict):
                formula = "".join(f"{el}{row[el]}" for el in sorted(row))
                formula_counter[formula] += 1
        summary["top_formulas"] = top_counter(formula_counter)

    if "metadata" in df.columns:
        provenance_counter: Counter[str] = Counter()
        substitution_counter: Counter[str] = Counter()
        sampled_counter: Counter[str] = Counter()
        for row in df["metadata"].dropna():
            metadata = normalize_metadata(row)
            if not metadata:
                continue
            if "provenance" in metadata:
                provenance_counter[str(metadata["provenance"])] += 1
            if "halide_substitution" in metadata:
                substitution_counter[str(metadata["halide_substitution"])] += 1
            for key in ["aimd", "sampled", "stuffed"]:
                if key in metadata:
                    sampled_counter[f"{key}={metadata[key]}"] += 1
        summary["top_provenance"] = top_counter(provenance_counter)
        summary["top_halide_substitution"] = top_counter(substitution_counter)
        summary["metadata_flags"] = top_counter(sampled_counter)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
