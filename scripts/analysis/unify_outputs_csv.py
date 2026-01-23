from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REQUIRED_COLS = [
    "tool",
    "tool_version",
    "platform",
    "format",
    "file_id",
    "window_s",
    "n_channels",
    "sequence",
    "overlay_state",
    "cache_state",
    "run_id",
    "step_id",
    "command_t_ms",
    "visible_t_ms",
    "latency_ms",
    "is_valid",
    "invalid_reason",
]

OPTIONAL_COLS = [
    "effective_window_s",
    "effective_n_channels",
    "viewport_t_start_s",
    "viewport_t_end_s",
    "notes",
    "gpu_info",
    "cpu_info",
    "os_version",
    "display_hz",
]

SUMMARY_COLS = [
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_max_ms",
    "updates_per_s",
]

ALL_COLS = REQUIRED_COLS + OPTIONAL_COLS + SUMMARY_COLS


TOOL_VERSION_KEYS = {
    "VIS_PG": "pyqtgraph",
    "VIS_VISPY": "vispy",
    "VIS_DATOVIZ": "datoviz",
    "VIS_FASTPLOTLIB": "fastplotlib",
    "VIS_PLOTLY": "plotly",
}


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_path(payload: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    cur: Any = payload
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def pick(payload: Dict[str, Any], paths: Iterable[Tuple[str, ...]]) -> Any:
    for path in paths:
        val = get_path(payload, path)
        if val is not None:
            return val
    return None


def to_float(value: Any) -> Any:
    if value is None or value == "":
        return ""
    try:
        return float(value)
    except Exception:
        return ""


def to_int(value: Any) -> Any:
    if value is None or value == "":
        return ""
    try:
        return int(float(value))
    except Exception:
        return ""


def to_bool_str(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "false"
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return "true"
    if s in {"0", "false", "no", "n"}:
        return "false"
    return "true"


def percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return float(vals[0])
    rank = (q / 100.0) * (len(vals) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(vals[lo])
    weight = rank - lo
    return float(vals[lo] + (vals[hi] - vals[lo]) * weight)


def find_runs(out_root: Path) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    for manifest_path in out_root.rglob("manifest.json"):
        if "_orchestration" in manifest_path.parts:
            continue
        yield manifest_path.parent, load_json(manifest_path)


def read_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def collect_env(manifest: Dict[str, Any]) -> Dict[str, Any]:
    env: Dict[str, Any] = {}
    for path in [("system",), ("env",), ("extra", "env")]:
        block = get_path(manifest, path)
        if isinstance(block, dict):
            env.update(block)
    return env


def tool_version(manifest: Dict[str, Any], tool_id: str) -> str:
    for path in [("tool_version",), ("extra", "tool_version")]:
        val = get_path(manifest, path)
        if isinstance(val, str) and val:
            return val
    packages = manifest.get("packages")
    if isinstance(packages, dict):
        key = TOOL_VERSION_KEYS.get(tool_id)
        if key and key in packages:
            return str(packages.get(key, ""))
    return ""


def normalize_common(
    manifest: Dict[str, Any],
    summary: Optional[Dict[str, Any]],
    bench_id: str,
    tool_id: str,
) -> Dict[str, Any]:
    env = collect_env(manifest)
    fmt = pick(
        manifest,
        [
            ("format",),
            ("args", "format"),
            ("params", "format"),
        ],
    )
    file_path = pick(
        manifest,
        [
            ("file",),
            ("args", "file"),
            ("params", "file"),
        ],
    )
    file_id = Path(str(file_path)).name if file_path else ""
    window_s = pick(
        manifest,
        [
            ("window_s",),
            ("args", "window_s"),
            ("params", "window_s"),
        ],
    )
    n_channels = pick(
        manifest,
        [
            ("n_channels",),
            ("args", "n_ch"),
            ("params", "n_channels"),
            ("params", "n_ch"),
        ],
    )
    sequence = pick(
        manifest,
        [
            ("sequence",),
            ("args", "sequence"),
            ("params", "sequence"),
        ],
    )
    overlay = pick(
        manifest,
        [
            ("overlay",),
            ("overlay_state",),
            ("args", "overlay"),
            ("args", "overlay_state"),
            ("params", "overlay"),
        ],
    )
    cache_state = pick(
        manifest,
        [
            ("cache_state",),
            ("params", "cache_state"),
            ("args", "cache_state"),
        ],
    )
    run_id = pick(
        manifest,
        [
            ("run_id",),
            ("args", "run_id"),
            ("params", "run_id"),
        ],
    )

    summary_meta = summary.get("meta") if isinstance(summary, dict) else {}
    effective_window_s = pick(
        summary or {},
        [
            ("effective_window_s",),
            ("meta", "effective_window_s"),
        ],
    )
    effective_n_channels = pick(
        summary or {},
        [
            ("effective_n_ch",),
            ("effective_n_channels",),
            ("meta", "effective_n_ch"),
            ("meta", "effective_n_channels"),
        ],
    )
    viewport_t_start_s = pick(
        summary or {},
        [
            ("viewport_t_start_s",),
            ("meta", "viewport_t_start_s"),
        ],
    )
    viewport_t_end_s = pick(
        summary or {},
        [
            ("viewport_t_end_s",),
            ("meta", "viewport_t_end_s"),
        ],
    )
    display_hz = pick(
        summary or {},
        [
            ("display_hz",),
            ("meta", "display_hz"),
        ],
    )
    notes = ""
    if isinstance(summary, dict):
        status = summary.get("status")
        error = summary.get("error")
        if status and str(status).upper() not in {"OK", "SUCCESS"}:
            notes = f"status={status}"
        if error:
            notes = f"{notes} error={error}".strip()
    if not notes and isinstance(summary_meta, dict):
        note = summary_meta.get("notes")
        if isinstance(note, str):
            notes = note

    platform = env.get("platform") or env.get("machine") or ""
    os_version = env.get("os_version") or ""
    cpu_info = env.get("processor") or env.get("cpu") or ""
    gpu_info = env.get("gpu") or env.get("graphics") or ""

    return {
        "tool": tool_id,
        "tool_version": tool_version(manifest, tool_id),
        "platform": str(platform) if platform is not None else "",
        "format": fmt or "",
        "file_id": file_id,
        "window_s": window_s if window_s is not None else "",
        "n_channels": n_channels if n_channels is not None else "",
        "sequence": sequence or "",
        "overlay_state": overlay or "",
        "cache_state": cache_state or "",
        "run_id": run_id if run_id is not None else "",
        "effective_window_s": effective_window_s if effective_window_s is not None else "",
        "effective_n_channels": effective_n_channels if effective_n_channels is not None else "",
        "viewport_t_start_s": viewport_t_start_s if viewport_t_start_s is not None else "",
        "viewport_t_end_s": viewport_t_end_s if viewport_t_end_s is not None else "",
        "notes": notes,
        "gpu_info": gpu_info if gpu_info is not None else "",
        "cpu_info": cpu_info if cpu_info is not None else "",
        "os_version": os_version if os_version is not None else "",
        "display_hz": display_hz if display_hz is not None else "",
    }


def pick_time(row: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in ("", None):
            return to_float(row[key])
    return ""


def infer_valid(row: Dict[str, Any]) -> Tuple[str, str]:
    status = str(row.get("status", "")).upper()
    if status in {"FAIL", "ERROR"}:
        return "false", "fail"
    if status in {"NOOP"}:
        return "false", "noop"
    noop_val = row.get("noop")
    if noop_val not in (None, "") and to_bool_str(noop_val) == "true":
        return "false", "noop"
    dropped = row.get("was_dropped")
    if dropped not in (None, "") and str(dropped).strip() == "1":
        return "false", "dropped"
    ok_val = row.get("ok")
    if ok_val not in (None, "") and to_bool_str(ok_val) == "false":
        return "false", "fail"
    return "true", ""


def build_step_rows(
    manifest: Dict[str, Any],
    summary: Optional[Dict[str, Any]],
    bench_id: str,
    tool_id: str,
    steps_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    common = normalize_common(manifest, summary, bench_id, tool_id)
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(steps_rows):
        step_id = row.get("step_id", idx)
        command_t_ms = pick_time(
            row,
            ["command_t_ms", "issued_ms", "t_cmd_ms", "cmd_ms", "issued_t_ms"],
        )
        visible_t_ms = pick_time(
            row,
            ["visible_t_ms", "presented_ms", "t_present_ms", "paint_ms"],
        )
        latency_ms = to_float(row.get("latency_ms", row.get("lat_ms", "")))
        if latency_ms == "" and row.get("latency_s") not in (None, ""):
            try:
                latency_ms = float(row.get("latency_s")) * 1000.0
            except Exception:
                latency_ms = ""
        if command_t_ms == "" and row.get("t_issue") not in (None, ""):
            try:
                command_t_ms = float(row.get("t_issue")) * 1000.0
            except Exception:
                command_t_ms = ""
        if visible_t_ms == "" and row.get("t_ack") not in (None, ""):
            try:
                visible_t_ms = float(row.get("t_ack")) * 1000.0
            except Exception:
                visible_t_ms = ""
        if latency_ms == "" and command_t_ms != "" and visible_t_ms != "":
            latency_ms = float(visible_t_ms) - float(command_t_ms)
        is_valid, invalid_reason = infer_valid(row)
        if isinstance(latency_ms, (int, float)) and math.isfinite(float(latency_ms)):
            if float(latency_ms) < 0:
                is_valid, invalid_reason = "false", "negative_latency"
        out = dict(common)
        out.update(
            {
                "step_id": to_int(step_id),
                "command_t_ms": command_t_ms,
                "visible_t_ms": visible_t_ms,
                "latency_ms": latency_ms,
                "is_valid": is_valid,
                "invalid_reason": invalid_reason,
            }
        )
        rows.append(out)
    return rows


def summary_from_steps(step_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    latencies: List[float] = []
    cmd_times: List[float] = []
    vis_times: List[float] = []
    for row in step_rows:
        if row.get("is_valid") != "true":
            continue
        lat = row.get("latency_ms")
        if isinstance(lat, (int, float)) and math.isfinite(float(lat)):
            if float(lat) >= 0:
                latencies.append(float(lat))
        cmd = row.get("command_t_ms")
        vis = row.get("visible_t_ms")
        if isinstance(cmd, (int, float)) and math.isfinite(float(cmd)):
            cmd_times.append(float(cmd))
        if isinstance(vis, (int, float)) and math.isfinite(float(vis)):
            vis_times.append(float(vis))

    p50 = percentile(latencies, 50)
    p95 = percentile(latencies, 95)
    pmax = max(latencies) if latencies else float("nan")

    updates_per_s = ""
    timebase = cmd_times if len(cmd_times) >= 2 else vis_times
    if len(timebase) >= 2:
        span_ms = max(timebase) - min(timebase)
        if span_ms > 0:
            updates_per_s = float((len(timebase) - 1) / (span_ms / 1000.0))

    return {
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "latency_max_ms": pmax,
        "updates_per_s": updates_per_s,
    }


def summary_row(
    manifest: Dict[str, Any],
    summary: Optional[Dict[str, Any]],
    bench_id: str,
    tool_id: str,
    step_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    common = normalize_common(manifest, summary, bench_id, tool_id)
    out = dict(common)
    out.update(
        {
            "step_id": -1,
            "command_t_ms": "",
            "visible_t_ms": "",
            "latency_ms": "",
            "is_valid": "true",
            "invalid_reason": "",
        }
    )

    if isinstance(summary, dict):
        status = summary.get("status")
        if status and str(status).upper() not in {"OK", "SUCCESS"}:
            out["is_valid"] = "false"
            out["invalid_reason"] = str(status)

    metrics = {}
    if isinstance(summary, dict):
        def _latency_ms_from_summary(ms_keys: List[Tuple[str, ...]], s_key: Tuple[str, ...]) -> Any:
            ms_val = pick(summary, ms_keys)
            if ms_val is not None and ms_val != "":
                return ms_val
            s_val = pick(summary, [s_key])
            if s_val is None or s_val == "":
                return None
            try:
                return float(s_val) * 1000.0
            except Exception:
                return None

        metrics = {
            "latency_p50_ms": _latency_ms_from_summary(
                [
                    ("lat_p50_ms",),
                    ("p50_ms",),
                    ("p50_latency_ms",),
                ],
                ("latency_p50_s",),
            ),
            "latency_p95_ms": _latency_ms_from_summary(
                [
                    ("lat_p95_ms",),
                    ("p95_ms",),
                    ("p95_latency_ms",),
                ],
                ("latency_p95_s",),
            ),
            "latency_max_ms": _latency_ms_from_summary(
                [
                    ("lat_max_ms",),
                    ("max_ms",),
                    ("latency_max_ms",),
                ],
                ("latency_max_s",),
            ),
            "updates_per_s": pick(
                summary,
                [
                    ("updates_per_s",),
                    ("throughput_ups",),
                    ("fps_presented",),
                    ("achieved_cmd_rate",),
                    ("achieved_frame_rate",),
                ],
            ),
        }

    derived = summary_from_steps(step_rows)
    for key in SUMMARY_COLS:
        val = metrics.get(key)
        if val is None or val == "":
            val = derived.get(key, "")
        out[key] = val if val is not None else ""

    # TFFR-only fallback
    if bench_id == "TFFR" and isinstance(summary, dict):
        tffr = summary.get("tffr_ms") or summary.get("first_render_ms")
        if tffr is not None:
            out["latency_p50_ms"] = float(tffr)
            out["latency_p95_ms"] = float(tffr)
            out["latency_max_ms"] = float(tffr)

    neg_metrics = []
    for key in ("latency_p50_ms", "latency_p95_ms", "latency_max_ms"):
        val = out.get(key)
        if isinstance(val, (int, float)) and math.isfinite(float(val)):
            if float(val) < 0:
                neg_metrics.append(key)
    if neg_metrics:
        out["is_valid"] = "false"
        out["invalid_reason"] = "negative_latency"

    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLS)
        writer.writeheader()
        for row in rows:
            out_row = {k: row.get(k, "") for k in ALL_COLS}
            writer.writerow(out_row)


def main() -> None:
    p = argparse.ArgumentParser(description="Unify benchmark outputs into per-tool/per-bench CSVs.")
    p.add_argument("--out-root", type=Path, default=Path("outputs"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/_aggregate/unified"))
    p.add_argument("--benches", nargs="*", default=["TFFR", "A1", "A2"])
    args = p.parse_args()

    out_root = args.out_root
    out_dir = args.out_dir

    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    for run_dir, manifest in find_runs(out_root):
        bench_id = manifest.get("bench_id")
        tool_id = manifest.get("tool_id") or manifest.get("tool")

        if not bench_id or not tool_id:
            parts = run_dir.parts
            try:
                idx = parts.index("outputs")
                bench_id = bench_id or parts[idx + 1]
                tool_id = tool_id or parts[idx + 2]
            except Exception:
                continue

        if bench_id not in args.benches:
            continue

        summary_path = run_dir / "summary.json"
        summary = load_json(summary_path) if summary_path.exists() else None

        steps_path = run_dir / "steps.csv"
        steps_rows = read_csv(steps_path) if steps_path.exists() else []

        step_rows = build_step_rows(manifest, summary, bench_id, tool_id, steps_rows) if steps_rows else []
        rows = step_rows[:]
        rows.append(summary_row(manifest, summary, bench_id, tool_id, step_rows))

        key = (tool_id, bench_id)
        buckets.setdefault(key, []).extend(rows)

    for (tool_id, bench_id), rows in buckets.items():
        out_path = out_dir / f"{tool_id}_{bench_id}.csv"
        write_csv(out_path, rows)


if __name__ == "__main__":
    main()
