from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


METRICS = [
    "latency_p50_ms_median",
    "latency_p95_ms_median",
    "latency_max_ms_median",
    "updates_per_s_median",
]


def infer_bench_id(path: Path, benches: Iterable[str]) -> str | None:
    name = path.name
    for bench_id in benches:
        suffix = f"_{bench_id}.csv"
        if name.endswith(suffix):
            return bench_id
    return None


def format_number(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.6g}"
    return str(value)


def write_markdown_table(path: Path, df: pd.DataFrame) -> None:
    cols = [format_number(c) for c in df.columns.tolist()]
    lines = []
    lines.append("| n_channels | " + " | ".join(cols) + " |")
    lines.append("|---|" + "|".join(["---"] * len(cols)) + "|")
    for idx, row in df.iterrows():
        values = [format_number(row[c]) for c in df.columns]
        lines.append("| " + format_number(idx) + " | " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_unified(
    unified_dir: Path, benches: Iterable[str]
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in unified_dir.glob("*.csv"):
        bench_id = infer_bench_id(path, benches)
        if bench_id is None:
            continue
        suffix = f"_{bench_id}.csv"
        tool_id = path.name[: -len(suffix)]
        df = pd.read_csv(path)
        if "step_id" not in df.columns:
            continue
        df = df[df["step_id"].astype(str).str.strip() == "-1"].copy()
        if df.empty:
            continue
        df["bench_id"] = bench_id
        if "tool" in df.columns:
            df["tool_id"] = df["tool"].fillna(tool_id)
        else:
            df["tool_id"] = tool_id
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_conditions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["window_s"] = pd.to_numeric(df["window_s"], errors="coerce")
    df["n_channels"] = pd.to_numeric(df["n_channels"], errors="coerce")
    df = df.dropna(subset=["window_s", "n_channels"])
    df["is_valid"] = df.get("is_valid", "").astype(str).str.lower()
    for metric in METRICS:
        base = metric.replace("_median", "")
        if base in df.columns:
            df[base] = pd.to_numeric(df[base], errors="coerce")
        else:
            df[base] = np.nan

    group_cols = [
        "bench_id",
        "tool_id",
        "window_s",
        "n_channels",
        "format",
        "sequence",
        "overlay_state",
        "cache_state",
    ]

    def _summarize(group: pd.DataFrame) -> pd.Series:
        valid = group[group["is_valid"] == "true"]
        out: Dict[str, object] = {
            "n_runs": len(group),
            "n_valid": len(valid),
        }
        for metric in METRICS:
            base = metric.replace("_median", "")
            vals = valid[base].dropna()
            out[metric] = float(vals.median()) if not vals.empty else np.nan
        return pd.Series(out)

    summary = (
        df.groupby(group_cols, dropna=False)
        .apply(_summarize)
        .reset_index()
    )
    return summary


def select_best_conditions(summary: pd.DataFrame) -> Tuple[pd.DataFrame, List[Tuple[str, str, int]]]:
    key_cols = ["bench_id", "tool_id", "window_s", "n_channels"]
    summary = summary.copy()

    summary = summary.sort_values(
        by=[
            "bench_id",
            "tool_id",
            "window_s",
            "n_channels",
            "n_valid",
            "n_runs",
            "format",
            "sequence",
            "overlay_state",
            "cache_state",
        ],
        ascending=[True, True, True, True, False, False, True, True, True, True],
    )

    dup_counts = (
        summary.groupby(key_cols)
        .size()
        .reset_index(name="condition_count")
    )
    multi = [
        (row["bench_id"], row["tool_id"], int(row["condition_count"]))
        for _, row in dup_counts.iterrows()
        if int(row["condition_count"]) > 1
    ]

    selected = summary.drop_duplicates(subset=key_cols, keep="first")
    return selected, multi


def write_matrices(
    selected: pd.DataFrame,
    out_root: Path,
) -> Tuple[int, List[Tuple[str, str, str]]]:
    produced = 0
    skipped: List[Tuple[str, str, str]] = []

    for (bench_id, tool_id), group in selected.groupby(["bench_id", "tool_id"]):
        group = group.copy()
        group["window_s"] = pd.to_numeric(group["window_s"], errors="coerce")
        group["n_channels"] = pd.to_numeric(group["n_channels"], errors="coerce")
        group = group.dropna(subset=["window_s", "n_channels"])
        if group.empty:
            skipped.append((bench_id, tool_id, "no_rows"))
            continue

        for metric in METRICS:
            pivot = group.pivot(index="n_channels", columns="window_s", values=metric)
            if pivot.empty or pivot.notna().sum().sum() == 0:
                skipped.append((bench_id, tool_id, metric))
                continue
            pivot = pivot.sort_index(axis=0).sort_index(axis=1)

            target_dir = out_root / bench_id / tool_id
            target_dir.mkdir(parents=True, exist_ok=True)
            csv_path = target_dir / f"{metric}.csv"
            md_path = target_dir / f"{metric}.md"

            pivot.to_csv(csv_path, index=True)
            write_markdown_table(md_path, pivot)
            produced += 1

    return produced, skipped


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Build window/channel matrices from unified outputs without mixed-condition averaging."
        )
    )
    p.add_argument("--unified-dir", type=Path, default=Path("outputs/_aggregate/unified"))
    p.add_argument("--out-dir", type=Path, default=Path("out/matrices"))
    p.add_argument(
        "--benches",
        nargs="*",
        default=["IO", "TFFR", "A1_THROUGHPUT", "A2_CADENCED", "A1", "A2"],
    )
    args = p.parse_args()

    df = read_unified(args.unified_dir, args.benches)
    if df.empty:
        print("No unified summary rows found; nothing to aggregate.")
        return 1

    summary = summarize_conditions(df)
    if summary.empty:
        print("No summary rows with window/channels found; nothing to aggregate.")
        return 1

    selected, multi = select_best_conditions(summary)
    produced, skipped = write_matrices(selected, args.out_dir)

    print(f"Produced {produced} matrices.")
    if multi:
        print(f"Resolved {len(multi)} multi-condition groups by n_valid/n_runs.")
    if skipped:
        print("Skipped due to missing data:")
        for bench_id, tool_id, reason in skipped:
            print(f"- {bench_id} / {tool_id}: {reason}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
