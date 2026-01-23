from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List


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
    return [r for r in rows if r.get("step_id") == "-1"]


def parse_floats(rows: Iterable[Dict[str, str]], key: str) -> List[float]:
    vals: List[float] = []
    for row in rows:
        val = row.get(key, "")
        try:
            fval = float(val)
        except Exception:
            continue
        if math.isfinite(fval):
            vals.append(fval)
    return vals


def aggregate_summary(summary_rows: List[Dict[str, str]]) -> Dict[str, object]:
    tool_id = summary_rows[0].get("tool") or summary_rows[0].get("tool_id") or ""
    bench_id = summary_rows[0].get("bench_id") or ""

    valid_rows = [r for r in summary_rows if r.get("is_valid") == "true"]
    out: Dict[str, object] = {
        "tool_id": tool_id,
        "bench_id": bench_id,
        "n_runs": len(summary_rows),
        "n_valid": len(valid_rows),
    }

    for metric in METRICS:
        vals = parse_floats(valid_rows, metric)
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
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def main() -> None:
    p = argparse.ArgumentParser(
        description="Aggregate unified A1/A2 CSVs into per-tool per-phase outputs and a combined CSV."
    )
    p.add_argument("--unified-dir", type=Path, default=Path("outputs/_aggregate/unified"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/_aggregate/by_tool_phase"))
    p.add_argument("--phases", nargs="*", default=["A1_THROUGHPUT", "A2_CADENCED"])
    p.add_argument("--all-name", default="all_tools_by_phase.csv")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, object]] = []
    for phase in args.phases:
        for path in args.unified_dir.glob(f"*_{phase}.csv"):
            rows = read_rows(path)
            summary_rows = select_summary_rows(rows)
            if not summary_rows:
                continue
            agg = aggregate_summary(summary_rows)
            if not agg.get("bench_id"):
                agg["bench_id"] = phase
            if not agg.get("tool_id"):
                agg["tool_id"] = path.name.split("_")[0]
            out_path = args.out_dir / f"{agg['tool_id']}_{agg['bench_id']}_aggregate.csv"
            write_csv(out_path, [agg])
            all_rows.append(agg)

    all_path = args.out_dir / args.all_name
    write_csv(all_path, all_rows)


if __name__ == "__main__":
    main()
