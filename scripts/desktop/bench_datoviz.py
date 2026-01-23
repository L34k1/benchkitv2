#!/usr/bin/env python3
"""scripts/desktop/bench_datoviz.py

Datoviz desktop benchmarks:
- TFFR: time-to-first-render (first frame callback)
- A1_THROUGHPUT: timer-driven viewport updates without waiting for ACK
- A2_CADENCED: cadenced viewport updates with per-step ACK via on_frame

Outputs (per bench):
  outputs/<BENCH_ID>/VIS_DATOVIZ/<tag>/
    - manifest.json
    - steps.csv (A1/A2)
    - summary.json
    - tffr.csv (TFFR)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchkit.common import ensure_dir, env_info, out_dir, utc_stamp, write_json
from benchkit.bench_defaults import (
    DEFAULT_STEPS,
    DEFAULT_TARGET_INTERVAL_MS,
    DEFAULT_WINDOW_S,
    default_load_duration_s,
)
from benchkit.datoviz_bench_utils import (
    StepSpec,
    build_step_rows,
    build_step_specs,
    summarize_latency_s,
)
from benchkit.lexicon import (
    BENCH_A1,
    BENCH_A2,
    BENCH_TFFR,
    CACHE_WARM,
    FMT_EDF,
    FMT_NWB,
    OVL_OFF,
    OVL_ON,
    SEQ_PAN,
    SEQ_PAN_ZOOM,
    SEQ_ZOOM_IN,
    SEQ_ZOOM_OUT,
    TOOL_DATOVIZ,
)
from benchkit.loaders import decimate_for_display, load_edf_segment_pyedflib, load_nwb_segment_pynwb

DEFAULT_WARMUP_FRAMES = 5
DEFAULT_STEP_TIMEOUT_S = 5.0
DEFAULT_HARD_TIMEOUT_S = 60.0
DEFAULT_TAIL_FRAMES = 1

STEP_FIELDS = [
    "step_index",
    "step_label",
    "t_issue",
    "t_ack",
    "latency_s",
    "xlim0",
    "xlim1",
    "ylim0",
    "ylim1",
    "status",
]


def normalize_bench_id(bench_id: str) -> str:
    if bench_id == "A1":
        return BENCH_A1
    if bench_id == "A2":
        return BENCH_A2
    return bench_id


def _tool_version() -> Optional[str]:
    try:
        from importlib.metadata import version

        return version("datoviz")
    except Exception:
        return None


def _write_manifest(out_base: Path, bench_id: str, args: argparse.Namespace, tool_version: Optional[str]) -> None:
    params = {
        "bench_id": bench_id,
        "format": args.format,
        "file": str(args.file),
        "window_s": float(args.window_s),
        "n_channels": int(args.n_ch),
        "sequence": args.sequence,
        "overlay": args.overlay,
        "runs": int(args.runs),
        "steps": int(args.steps),
        "target_interval_ms": float(args.target_interval_ms),
        "step_timeout_s": float(args.step_timeout_s),
        "load_start_s": float(args.load_start_s),
        "load_duration_s": float(args.load_duration_s),
        "max_points_per_trace": int(args.max_points_per_trace),
        "width": int(args.width),
        "height": int(args.height),
        "warmup_frames": int(args.warmup_frames),
        "hard_timeout_s": float(args.hard_timeout_s),
        "cache_state": args.cache_state,
        "nwb_series_path": args.nwb_series_path,
        "nwb_time_dim": args.nwb_time_dim,
    }
    manifest = {
        "tool": "datoviz",
        "tool_id": TOOL_DATOVIZ,
        "tool_version": tool_version,
        "bench_id": bench_id,
        "format": args.format,
        "file": str(args.file),
        "window_s": float(args.window_s),
        "n_channels": int(args.n_ch),
        "sequence": args.sequence,
        "overlay": args.overlay,
        "run_id": 0,
        "steps_target": int(args.steps),
        "timestamp_utc": utc_stamp(),
        "params": params,
        "env_flags": {"vsync": args.vsync},
        "system": env_info(),
    }
    write_json(out_base / "manifest.json", manifest)


def _write_steps_csv(out_base: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path = out_base / "steps.csv"
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        import csv

        w = csv.DictWriter(f, fieldnames=STEP_FIELDS)
        w.writeheader()
        for row in rows:
            clean = {k: row.get(k) for k in STEP_FIELDS}
            w.writerow(clean)


def _write_summary(out_base: Path, summary: Dict[str, Any]) -> None:
    write_json(out_base / "summary.json", summary)


def _write_tffr_csv(out_base: Path, tffr_s: Optional[float]) -> None:
    path = out_base / "tffr.csv"
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        import csv

        w = csv.DictWriter(f, fieldnames=["run_id", "tffr_ms"])
        w.writeheader()
        tffr_ms = float(tffr_s) * 1000.0 if tffr_s is not None else float("nan")
        w.writerow({"run_id": 0, "tffr_ms": tffr_ms})


def _load_segment(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], float]:
    if args.format == FMT_EDF:
        seg = load_edf_segment_pyedflib(args.file, args.load_start_s, args.load_duration_s, args.n_ch)
    elif args.format == FMT_NWB:
        seg = load_nwb_segment_pynwb(
            args.file,
            args.load_start_s,
            args.load_duration_s,
            args.n_ch,
            series_path=args.nwb_series_path,
            time_dim=args.nwb_time_dim,
        )
    else:
        raise ValueError(f"Unsupported format: {args.format}")

    t, d, dec = decimate_for_display(seg.times_s, seg.data, args.max_points_per_trace)
    meta = dict(seg.meta)
    meta["decim_factor"] = int(dec)
    meta["n_points_per_trace"] = int(t.shape[0])
    meta["fs_hz"] = float(seg.fs_hz)
    return t.astype(np.float32), d.astype(np.float32), meta, float(seg.fs_hz)


def _safe_stop(app: Any) -> None:
    if app is None:
        return
    for name in ("stop", "quit"):
        if hasattr(app, name):
            try:
                getattr(app, name)()
                return
            except Exception:
                pass


def _safe_destroy(app: Any) -> None:
    if app is None:
        return
    if hasattr(app, "destroy"):
        try:
            app.destroy()
        except Exception:
            pass


def _safe_close(canvas: Any) -> None:
    if canvas is None:
        return
    if hasattr(canvas, "close"):
        try:
            canvas.close()
        except Exception:
            pass


def _register_on_frame(app: Any, canvas: Any, cb) -> None:
    if not hasattr(app, "on_frame"):
        raise RuntimeError("Datoviz app.on_frame is required for ACK callbacks.")
    try:
        @app.on_frame(canvas)
        def _on_frame(_ev) -> None:  # noqa: N801
            cb()
    except TypeError:
        try:
            @app.on_frame()
            def _on_frame(_ev) -> None:  # noqa: N801
                cb()
        except TypeError as exc:
            raise RuntimeError("Unsupported Datoviz app.on_frame signature.") from exc


def _register_timer(app: Any, interval_s: float, cb) -> None:
    if not hasattr(app, "timer"):
        raise RuntimeError("Datoviz app.timer is required for scheduling.")
    try:
        @app.timer(period=interval_s)
        def _on_timer(_ev) -> None:  # noqa: N801
            cb()
    except TypeError:
        try:
            @app.timer(delay=0.0, period=interval_s)
            def _on_timer(_ev) -> None:  # noqa: N801
                cb()
        except TypeError as exc:
            raise RuntimeError("Unsupported Datoviz app.timer signature.") from exc


def _make_canvas(app: Any, width: int, height: int) -> Any:
    if hasattr(app, "canvas"):
        try:
            return app.canvas(width=width, height=height, show=True)
        except TypeError:
            try:
                return app.canvas(size=(width, height), show=True)
            except TypeError:
                return app.canvas((width, height))
    if hasattr(app, "figure"):
        try:
            return app.figure(width=width, height=height, gui=True)
        except TypeError:
            return app.figure(width=width, height=height)
    raise RuntimeError("Datoviz App lacks canvas()/figure().")


def _set_line_pos(line: Any, pos: np.ndarray) -> None:
    if hasattr(line, "set_position"):
        try:
            line.set_position(pos)
            return
        except Exception:
            pass
    if hasattr(line, "set_data"):
        try:
            line.set_data(position=pos)
            return
        except Exception:
            pass
    if hasattr(line, "data"):
        try:
            line.data("pos", pos)
            return
        except Exception:
            pass
    elif hasattr(line, "set"):
        try:
            line.set(pos=pos)
        except Exception:
            pass


def _normalize_pos(axes: Any, pos: np.ndarray) -> np.ndarray:
    pos64 = np.asarray(pos, dtype=np.float64)
    pos_tr = axes.normalize(pos64)
    return np.asarray(pos_tr, dtype=np.float32)


def _make_scene(app: Any, t: np.ndarray, data: np.ndarray, window_s: float, overlay: str, width: int, height: int):
    canvas = _make_canvas(app, width, height)
    panel = canvas.panel()

    lo = float(t[0])
    hi = float(t[-1])
    x0 = lo
    x1 = min(hi, lo + float(window_s))

    offsets = np.arange(data.shape[0], dtype=np.float32)[:, None]
    y = data + offsets
    y0 = float(np.min(y)) - 0.5
    y1 = float(np.max(y)) + 0.5

    if not hasattr(panel, "axes"):
        raise RuntimeError("Datoviz panel.axes() not available.")
    axes = panel.axes((x0, x1), (y0, y1))
    if not (hasattr(axes, "xlim") and hasattr(axes, "ylim")):
        raise RuntimeError("Datoviz axes missing xlim/ylim.")

    x = t[None, :].repeat(y.shape[0], axis=0)
    visuals: List[Tuple[Any, np.ndarray]] = []
    for ch in range(y.shape[0]):
        pos = np.column_stack([x[ch], y[ch]]).astype(np.float32, copy=False)
        if not hasattr(app, "basic"):
            raise RuntimeError("Datoviz basic() API missing.")
        color = np.tile(np.array([255, 255, 255, 255], dtype=np.uint8), (pos.shape[0], 1))
        pos_tr = _normalize_pos(axes, pos)
        line = app.basic(topology="line_strip", position=pos_tr, color=color)
        panel.add(line)
        visuals.append((line, pos))

    ruler = None
    if overlay == OVL_ON:
        for frac in (0.25, 0.5, 0.75):
            x_line = x0 + (x1 - x0) * frac
            pos = np.array([[x_line, y0], [x_line, y1]], dtype=np.float32)
            if not hasattr(app, "basic"):
                raise RuntimeError("Datoviz overlay requires basic().")
            color = np.tile(np.array([153, 153, 153, 255], dtype=np.uint8), (pos.shape[0], 1))
            pos_tr = _normalize_pos(axes, pos)
            line = app.basic(topology="line_strip", position=pos_tr, color=color)
            panel.add(line)
            visuals.append((line, pos))
        # Moving ruler for visual confirmation of viewport changes.
        color = np.tile(np.array([255, 0, 0, 255], dtype=np.uint8), (2, 1))
        pos_r = np.array([[x0, y0], [x0, y1]], dtype=np.float32)
        ruler = app.basic(topology="line_strip", position=_normalize_pos(axes, pos_r), color=color)
        panel.add(ruler)

    return canvas, panel, axes, (y0, y1), (lo, hi), visuals, ruler


def _run_app(app: Any) -> None:
    if not hasattr(app, "run"):
        raise RuntimeError("Datoviz App missing run().")
    app.run()


def _apply_view(
    axes: Any,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    canvas: Any,
    visuals: List[Tuple[Any, np.ndarray]],
    ruler: Optional[Any],
) -> None:
    axes.xlim(float(x0), float(x1))
    axes.ylim(float(y0), float(y1))
    for line, pos in visuals:
        _set_line_pos(line, _normalize_pos(axes, pos))
    if ruler is not None:
        x_mid = float(x0 + 0.5 * (x1 - x0))
        pos = np.array([[x_mid, y0], [x_mid, y1]], dtype=np.float32)
        _set_line_pos(ruler, _normalize_pos(axes, pos))
    if hasattr(canvas, "update"):
        canvas.update()


def _run_tffr(
    app: Any,
    canvas: Any,
    axes: Any,
    ylims: Tuple[float, float],
    ruler: Optional[Any],
    visuals: List[Tuple[Any, np.ndarray]],
    t_start: float,
    x0: float,
    x1: float,
    hard_timeout_s: float,
) -> Optional[float]:
    done = {"flag": False, "tffr": None}
    wall_start = time.perf_counter()

    def on_frame() -> None:
        if done["flag"]:
            return
        now = time.perf_counter()
        done["flag"] = True
        done["tffr"] = now - t_start
        _safe_stop(app)

    def on_timer() -> None:
        if done["flag"]:
            return
        if time.perf_counter() - wall_start > hard_timeout_s:
            done["flag"] = True
            _safe_stop(app)

    _register_on_frame(app, canvas, on_frame)
    _register_timer(app, min(0.01, hard_timeout_s), on_timer)

    _apply_view(axes, x0, x1, ylims[0], ylims[1], canvas, visuals, ruler)
    _run_app(app)
    return done["tffr"]


def _run_a2(
    app: Any,
    canvas: Any,
    axes: Any,
    step_specs: List[StepSpec],
    ylims: Tuple[float, float],
    ruler: Optional[Any],
    visuals: List[Tuple[Any, np.ndarray]],
    cadence_s: float,
    warmup_frames: int,
    step_timeout_s: float,
    hard_timeout_s: float,
    tail_frames: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    issue_times: List[Optional[float]] = [None] * len(step_specs)
    ack_times: List[Optional[float]] = [None] * len(step_specs)
    statuses: List[str] = ["PENDING"] * len(step_specs)

    frames_seen = 0
    frames_after_start = 0
    started = False
    start_time = None
    end_time = None
    pending_idx: Optional[int] = None
    next_idx = 0
    next_due = None
    fatal_error = {"flag": False, "reason": None}
    last_frame_index = -1

    wall_start = time.perf_counter()
    interval_s = min(max(cadence_s, 0.001), 0.05)

    def on_frame() -> None:
        nonlocal frames_seen, frames_after_start, started, start_time, pending_idx, last_frame_index
        frames_seen += 1
        if frames_seen <= warmup_frames:
            return
        if not started:
            started = True
            start_time = time.perf_counter()
        frames_after_start += 1
        if pending_idx is not None:
            now = time.perf_counter()
            ack_times[pending_idx] = now
            statuses[pending_idx] = "OK"
            pending_idx = None
            last_frame_index = frames_after_start
            if next_idx >= len(step_specs):
                _safe_stop(app)

    def on_timer() -> None:
        nonlocal pending_idx, next_idx, next_due, end_time, started, start_time
        now = time.perf_counter()
        if now - wall_start > hard_timeout_s:
            fatal_error["flag"] = True
            fatal_error["reason"] = "hard_timeout"
            _safe_stop(app)
            return
        if not started:
            if warmup_frames <= 0 and start_time is None:
                start_time = now
                started = True
            elif hasattr(canvas, "update"):
                canvas.update()
            return
        if next_due is None:
            next_due = start_time
        if pending_idx is not None:
            if issue_times[pending_idx] is not None and (now - issue_times[pending_idx]) > step_timeout_s:
                statuses[pending_idx] = "FAIL"
                fatal_error["flag"] = True
                fatal_error["reason"] = "step_timeout"
                _safe_stop(app)
            return
        if next_idx >= len(step_specs):
            if end_time is None:
                end_time = now
            if frames_after_start - last_frame_index >= tail_frames:
                _safe_stop(app)
            return
        if now < float(next_due):
            return
        spec = step_specs[next_idx]
        issue_times[next_idx] = now
        statuses[next_idx] = "PENDING"
        pending_idx = next_idx
        next_idx += 1
        next_due = (start_time or now) + next_idx * cadence_s
        _apply_view(axes, spec.x0, spec.x1, ylims[0], ylims[1], canvas, visuals, ruler)

    _register_on_frame(app, canvas, on_frame)
    _register_timer(app, interval_s, on_timer)
    if hasattr(canvas, "update"):
        canvas.update()
    _run_app(app)

    if fatal_error["flag"]:
        for idx in range(len(step_specs)):
            if statuses[idx] == "PENDING":
                statuses[idx] = "SKIP"

    rows = build_step_rows(step_specs, issue_times, ack_times, statuses)
    elapsed_s = float("nan")
    if start_time is not None:
        end_time = end_time or time.perf_counter()
        elapsed_s = end_time - start_time

    summary: Dict[str, Any] = {
        "bench_id": BENCH_A2,
        "tool_id": TOOL_DATOVIZ,
        "steps": int(len(step_specs)),
        "count": int(len(step_specs)),
        "frames_presented": int(frames_after_start),
        "target_interval_ms": float(cadence_s * 1000.0),
        "step_timeout_s": float(step_timeout_s),
        "elapsed_s": float(elapsed_s),
        "status": "OK" if not fatal_error["flag"] else "ERROR",
    }
    if fatal_error["flag"]:
        summary["error"] = str(fatal_error["reason"])
    latencies = [r["latency_s"] for r in rows if r["status"] == "OK" and r["latency_s"] is not None]
    summary.update(summarize_latency_s(latencies))
    success = sum(1 for r in rows if r["status"] == "OK")
    summary["success_count"] = int(success)
    summary["success_rate"] = float(success) / float(len(step_specs)) if step_specs else 0.0
    return rows, summary


def _run_a1(
    app: Any,
    canvas: Any,
    axes: Any,
    step_specs: List[StepSpec],
    ylims: Tuple[float, float],
    ruler: Optional[Any],
    visuals: List[Tuple[Any, np.ndarray]],
    cadence_s: float,
    warmup_frames: int,
    hard_timeout_s: float,
    tail_frames: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    issue_times: List[Optional[float]] = [None] * len(step_specs)
    ack_times: List[Optional[float]] = [None] * len(step_specs)
    statuses: List[str] = ["PENDING"] * len(step_specs)

    frames_seen = 0
    frames_after_start = 0
    started = False
    start_time = None
    end_time = None
    next_idx = 0
    next_due = None
    last_frame_issued = -1
    coalesced_counts: List[int] = []
    frame_count_at_end = None
    fatal_error = {"flag": False, "reason": None}

    wall_start = time.perf_counter()
    interval_s = min(max(cadence_s, 0.001), 0.05)

    def on_frame() -> None:
        nonlocal frames_seen, frames_after_start, started, start_time, last_frame_issued
        frames_seen += 1
        if frames_seen <= warmup_frames:
            return
        now = time.perf_counter()
        if not started:
            started = True
            start_time = now
        frames_after_start += 1
        if next_idx > 0:
            latest = next_idx - 1
            if latest > last_frame_issued:
                for idx in range(last_frame_issued + 1, latest + 1):
                    if ack_times[idx] is None:
                        ack_times[idx] = now
                        statuses[idx] = "OK"
                coalesced_counts.append(latest - last_frame_issued)
                last_frame_issued = latest
        if end_time is not None and frame_count_at_end is not None:
            if frames_after_start - frame_count_at_end >= tail_frames:
                _safe_stop(app)

    def on_timer() -> None:
        nonlocal next_idx, next_due, end_time, frame_count_at_end, started, start_time
        now = time.perf_counter()
        if now - wall_start > hard_timeout_s:
            fatal_error["flag"] = True
            fatal_error["reason"] = "hard_timeout"
            _safe_stop(app)
            return
        if not started:
            if warmup_frames <= 0 and start_time is None:
                start_time = now
                started = True
            elif hasattr(canvas, "update"):
                canvas.update()
            return
        if next_due is None:
            next_due = start_time
        if next_idx >= len(step_specs):
            if end_time is None:
                end_time = now
                frame_count_at_end = frames_after_start
            return
        if now < float(next_due):
            return
        spec = step_specs[next_idx]
        issue_times[next_idx] = now
        statuses[next_idx] = "PENDING"
        _apply_view(axes, spec.x0, spec.x1, ylims[0], ylims[1], canvas, visuals, ruler)
        next_idx += 1
        next_due = (start_time or now) + next_idx * cadence_s

    _register_on_frame(app, canvas, on_frame)
    _register_timer(app, interval_s, on_timer)
    if hasattr(canvas, "update"):
        canvas.update()
    _run_app(app)

    for idx in range(len(step_specs)):
        if statuses[idx] == "PENDING":
            statuses[idx] = "NO_ACK" if issue_times[idx] is not None else "SKIP"

    rows = build_step_rows(step_specs, issue_times, ack_times, statuses)
    elapsed_s = float("nan")
    if start_time is not None:
        end_time = end_time or time.perf_counter()
        elapsed_s = end_time - start_time

    summary: Dict[str, Any] = {
        "bench_id": BENCH_A1,
        "tool_id": TOOL_DATOVIZ,
        "steps": int(len(step_specs)),
        "attempted_cmd_count": int(next_idx),
        "frame_count": int(frames_after_start),
        "elapsed_s": float(elapsed_s),
        "ack_policy": "nearest_frame",
        "status": "OK" if not fatal_error["flag"] else "ERROR",
    }
    if fatal_error["flag"]:
        summary["error"] = str(fatal_error["reason"])
    if elapsed_s and np.isfinite(elapsed_s) and elapsed_s > 0:
        summary["achieved_cmd_rate"] = float(next_idx) / float(elapsed_s)
        summary["achieved_frame_rate"] = float(frames_after_start) / float(elapsed_s)
    if coalesced_counts:
        summary["coalesced_max"] = int(max(coalesced_counts))
        summary["coalesced_mean"] = float(np.mean(coalesced_counts))
    return rows, summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Datoviz desktop benchmarks (TFFR/A1/A2).")
    ap.add_argument("--bench-id", type=str, required=True, choices=[BENCH_TFFR, BENCH_A1, BENCH_A2, "A1", "A2"])
    ap.add_argument("--format", choices=[FMT_EDF, FMT_NWB], required=True)
    ap.add_argument("--file", type=Path, required=True)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--n-ch", type=int, default=16)
    ap.add_argument("--load-start-s", type=float, default=0.0)
    ap.add_argument("--load-duration-s", type=float, default=None)
    ap.add_argument("--max-points-per-trace", type=int, default=5000)
    ap.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S)
    ap.add_argument("--sequence", choices=[SEQ_PAN, SEQ_ZOOM_IN, SEQ_ZOOM_OUT, SEQ_PAN_ZOOM], default=SEQ_PAN_ZOOM)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--target-interval-ms", type=float, default=DEFAULT_TARGET_INTERVAL_MS)
    ap.add_argument("--step-timeout-s", type=float, default=DEFAULT_STEP_TIMEOUT_S)
    ap.add_argument("--hard-timeout-s", type=float, default=DEFAULT_HARD_TIMEOUT_S)
    ap.add_argument("--warmup-frames", type=int, default=DEFAULT_WARMUP_FRAMES)
    ap.add_argument("--tail-frames", type=int, default=DEFAULT_TAIL_FRAMES)
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--height", type=int, default=700)
    ap.add_argument("--out-root", type=Path, default=Path("outputs"))
    ap.add_argument("--tag", type=str, default="datoviz")
    ap.add_argument("--overlay", choices=[OVL_OFF, OVL_ON], default=OVL_OFF)
    ap.add_argument("--cache-state", type=str, default=CACHE_WARM)
    ap.add_argument("--nwb-series-path", type=str, default=None)
    ap.add_argument("--nwb-time-dim", type=str, default="auto", choices=["auto", "time_first", "time_last"])
    ap.add_argument("--vsync", type=str, default="unknown", choices=["on", "off", "unknown"])
    args = ap.parse_args()

    args.bench_id = normalize_bench_id(args.bench_id)
    if args.load_duration_s is None:
        args.load_duration_s = default_load_duration_s(args.window_s)

    if args.runs != 1:
        print(f"[WARN] forcing runs=1 (requested {args.runs})")
        args.runs = 1

    out_base = out_dir(Path(args.out_root), args.bench_id, TOOL_DATOVIZ, args.tag)
    tool_version = _tool_version()
    _write_manifest(out_base, args.bench_id, args, tool_version)

    try:
        import datoviz as dv  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        print(f"[ERROR] datoviz not installed: {exc}")
        return 1

    t, data, meta, fs = _load_segment(args)
    app = None
    canvas = None
    axes = None
    ruler = None
    visuals: List[Tuple[Any, np.ndarray]] = []
    ylims: Tuple[float, float]
    xlims: Tuple[float, float]
    try:
        if not hasattr(dv, "App"):
            raise RuntimeError("Datoviz App() not available.")
        app = dv.App()
        canvas, _panel, axes, ylims, xlims, visuals, ruler = _make_scene(
            app,
            t,
            data,
            float(args.window_s),
            args.overlay,
            int(args.width),
            int(args.height),
        )

        if args.bench_id == BENCH_TFFR:
            t_start = time.perf_counter()
            tffr_s = _run_tffr(
                app,
                canvas,
                axes,
                ylims,
                ruler,
                visuals,
                t_start,
                xlims[0],
                xlims[0] + float(args.window_s),
                float(args.hard_timeout_s),
            )
            _write_tffr_csv(out_base, tffr_s)
            summary = {
                "bench_id": BENCH_TFFR,
                "tool_id": TOOL_DATOVIZ,
                "format": args.format,
                "window_s": float(args.window_s),
                "n_channels": int(args.n_ch),
                "tffr_s": float(tffr_s) if tffr_s is not None else None,
                "tffr_ms": float(tffr_s) * 1000.0 if tffr_s is not None else float("nan"),
                "overlay": args.overlay,
                "meta": meta,
                "status": "OK" if tffr_s is not None else "ERROR",
            }
            if tffr_s is None:
                summary["error"] = "timeout"
                _write_summary(out_base, summary)
                return 1
            _write_summary(out_base, summary)
            return 0

        step_specs = build_step_specs(
            args.sequence,
            xlims[0],
            xlims[1],
            float(args.window_s),
            int(args.steps),
            ylims[0],
            ylims[1],
        )
        cadence_s = float(args.target_interval_ms) / 1000.0

        if args.bench_id == BENCH_A2:
            rows, summary = _run_a2(
                app,
                canvas,
                axes,
                step_specs,
                ylims,
                ruler,
                visuals,
                cadence_s,
                int(args.warmup_frames),
                float(args.step_timeout_s),
                float(args.hard_timeout_s),
                int(args.tail_frames),
            )
        else:
            rows, summary = _run_a1(
                app,
                canvas,
                axes,
                step_specs,
                ylims,
                ruler,
                visuals,
                cadence_s,
                int(args.warmup_frames),
                float(args.hard_timeout_s),
                int(args.tail_frames),
            )

        summary.update(
            {
                "format": args.format,
                "sequence": args.sequence,
                "overlay": args.overlay,
                "window_s": float(args.window_s),
                "n_channels": int(args.n_ch),
                "target_interval_ms": float(args.target_interval_ms),
                "meta": meta,
            }
        )
        _write_steps_csv(out_base, rows)
        _write_summary(out_base, summary)

        if summary.get("status") != "OK":
            return 1
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}")
        _write_summary(
            out_base,
            {
                "bench_id": args.bench_id,
                "tool_id": TOOL_DATOVIZ,
                "status": "ERROR",
                "error": str(exc),
                "format": args.format,
            },
        )
        return 1
    finally:
        _safe_stop(app)
        _safe_close(canvas)
        _safe_destroy(app)


if __name__ == "__main__":
    raise SystemExit(main())
