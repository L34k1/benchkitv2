from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


METRICS = [
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_max_ms",
    "updates_per_s",
]


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def select_summary_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    return [r for r in rows if str(r.get("step_id", "")).strip() == "-1"]


def parse_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        val = float(value)
    except Exception:
        return None
    if not math.isfinite(val):
        return None
    return val


def parse_key(value: object) -> str:
    return "" if value is None else str(value)


def infer_bench_id(path: Path, benches: Iterable[str]) -> str | None:
    name = path.name
    for bench_id in benches:
        suffix = f"_{bench_id}.csv"
        if name.endswith(suffix):
            return bench_id
    return None


def add_unique(values: Set[str], value: object) -> None:
    sval = parse_key(value).strip()
    if sval:
        values.add(sval)


def aggregate_group(rows: List[Dict[str, str]]) -> Dict[str, object]:
    tool_id = parse_key(rows[0].get("tool") or rows[0].get("tool_id"))
    window_s = parse_key(rows[0].get("window_s"))
    n_channels = parse_key(rows[0].get("n_channels"))
    bench_id = parse_key(rows[0].get("bench_id"))

    formats: Set[str] = set()
    sequences: Set[str] = set()
    overlays: Set[str] = set()
    caches: Set[str] = set()

    valid_rows = [r for r in rows if str(r.get("is_valid", "")).lower() == "true"]

    for row in rows:
        add_unique(formats, row.get("format"))
        add_unique(sequences, row.get("sequence"))
        add_unique(overlays, row.get("overlay_state"))
        add_unique(caches, row.get("cache_state"))

    out: Dict[str, object] = {
        "tool_id": tool_id,
        "bench_id": bench_id,
        "window_s": window_s,
        "n_channels": n_channels,
        "n_runs": len(rows),
        "n_valid": len(valid_rows),
        "formats": "|".join(sorted(formats)),
        "sequences": "|".join(sorted(sequences)),
        "overlay_states": "|".join(sorted(overlays)),
        "cache_states": "|".join(sorted(caches)),
    }

    for metric in METRICS:
        vals = [parse_float(r.get(metric)) for r in valid_rows]
        vals = [v for v in vals if v is not None]
        if not vals:
            out[f"{metric}_mean"] = ""
            out[f"{metric}_median"] = ""
            out[f"{metric}_min"] = ""
            out[f"{metric}_max"] = ""
            out[f"{metric}_std"] = ""
            continue
        out[f"{metric}_mean"] = statistics.fmean(vals)
        out[f"{metric}_median"] = statistics.median(vals)
        out[f"{metric}_min"] = min(vals)
        out[f"{metric}_max"] = max(vals)
        out[f"{metric}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0

    return out


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Aggregate unified CSVs into per-tool/window/channel summaries to compare scaling effects."
        )
    )
    p.add_argument("--unified-dir", type=Path, default=Path("outputs/_aggregate/unified"))
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/_aggregate/by_tool_window_channels"),
    )
    p.add_argument(
        "--benches",
        nargs="*",
        default=["IO", "TFFR", "A1_THROUGHPUT", "A2_CADENCED", "A1", "A2"],
    )
    p.add_argument("--all-name", default="all_tools_by_window_channels.csv")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, object]] = []
    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, str]]] = {}

    for path in args.unified_dir.glob("*.csv"):
        bench_id = infer_bench_id(path, args.benches)
        if bench_id is None:
            continue
        rows = select_summary_rows(read_rows(path))
        for row in rows:
            tool_id = parse_key(row.get("tool") or row.get("tool_id"))
            window_s = parse_key(row.get("window_s"))
            n_channels = parse_key(row.get("n_channels"))
            if not tool_id or window_s == "" or n_channels == "":
                continue
            row["bench_id"] = bench_id
            key = (tool_id, bench_id, window_s, n_channels)
            grouped.setdefault(key, []).append(row)

    per_tool_bench: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for (tool_id, bench_id, window_s, n_channels), rows in grouped.items():
        agg = aggregate_group(rows)
        all_rows.append(agg)
        per_tool_bench.setdefault((tool_id, bench_id), []).append(agg)

    for (tool_id, bench_id), rows in per_tool_bench.items():
        def _sort_key(row: Dict[str, object]) -> Tuple[float, float]:
            win = parse_float(row.get("window_s"))
            nch = parse_float(row.get("n_channels"))
            return (win if win is not None else float("inf"), nch if nch is not None else float("inf"))

        rows = sorted(rows, key=_sort_key)
        out_path = args.out_dir / f"{tool_id}_{bench_id}_window_channels.csv"
        write_csv(out_path, rows)

    all_path = args.out_dir / args.all_name
    write_csv(all_path, sorted(all_rows, key=lambda r: (r.get("tool_id", ""), r.get("bench_id", ""))))


if __name__ == "__main__":
    main()
