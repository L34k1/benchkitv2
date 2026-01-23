#!/usr/bin/env python3
"""fastplotlib A2 benchmark: throughput-only pan/zoom submissions."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from _fastplotlib_common import (
    ack_frame,
    apply_step,
    bench_json_line,
    build_figure,
    load_data,
    make_sequence,
    request_draw,
    safe_close,
    show_figure,
    summarize_values,
    write_manifest,
    write_steps_csv,
    write_summary,
)


def _initial_view(times: np.ndarray, data: np.ndarray, window_s: float) -> Dict[str, Any]:
    x_min = float(times[0])
    x_max = min(float(times[-1]), x_min + float(window_s))
    gain = 1.0
    spacing = 2.0
    offsets = np.arange(data.shape[0], dtype=np.float32) * spacing
    y = data * gain + offsets[:, None]
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "gain": gain,
        "spacing": spacing,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="fastplotlib A2 throughput-only benchmark.")
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--fs-hz", type=float, default=-1.0)
    p.add_argument("--window-s", type=float, required=True)
    p.add_argument("--n-ch", type=int, required=True)
    p.add_argument("--seq", choices=["PAN", "ZOOM_IN", "ZOOM_OUT", "PAN_ZOOM"], required=True)
    p.add_argument("--n-steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--backend", choices=["auto", "qt", "glfw", "offscreen"], default="auto")
    p.add_argument("--canvas-vsync", type=int, choices=[0, 1], default=1)
    p.add_argument("--canvas-max-fps", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    outdir = args.outdir
    status = 0
    fig = None
    summary: Dict[str, Any] = {}

    try:
        times, data, fs_hz, meta = load_data(
            args.file,
            args.fs_hz,
            args.window_s,
            args.n_ch,
        )
        initial_view = _initial_view(times, data, args.window_s)
        canvas_kwargs = {"vsync": bool(args.canvas_vsync)}
        if int(args.canvas_max_fps) > 0:
            canvas_kwargs["max_fps"] = int(args.canvas_max_fps)
        backend_opts = {"backend": args.backend, "canvas_kwargs": canvas_kwargs}
        fig, subplot, cam_state, backend_info = build_figure(data, times, fs_hz, initial_view, backend_opts)
        initial_view["viewport_rect"] = tuple(float(x) for x in subplot.viewport.rect)

        write_manifest(outdir, vars(args), initial_view, cam_state)

        if not show_figure(fig):
            summary = {
                "bench": "A2",
                "tool": "fastplotlib",
                "status": "ERROR",
                "error": "show_failed",
            }
            write_summary(outdir, summary)
            bench_json_line(summary)
            return 1

        ack_state: Dict[str, Any] = {}
        request_draw(fig)
        if not ack_frame(fig, 2.0, "auto", ack_state):
            summary = {
                "bench": "A2",
                "tool": "fastplotlib",
                "status": "ERROR",
                "error": "warmup_timeout",
            }
            write_summary(outdir, summary)
            bench_json_line(summary)
            return 1

        seq = make_sequence(args.seq, args.n_steps, args.seed, initial_view, {})
        rows: List[Dict[str, Any]] = []
        t_start = time.perf_counter()
        for idx, step in enumerate(seq):
            t0 = time.perf_counter()
            apply_step(subplot, step)
            request_draw(fig)
            t1 = time.perf_counter()
            submit_ms = (t1 - t0) * 1000.0
            rows.append(
                {
                    "step_idx": idx,
                    "op": step.op,
                    "world_dx": float(step.world_dx),
                    "zoom_factor": float(step.zoom_factor) if step.zoom_factor is not None else None,
                    "submit_ms": float(submit_ms),
                    "ok": True,
                    "timeout": False,
                    "tool": "fastplotlib",
                    "bench": "A2",
                    "window_s": float(args.window_s),
                    "n_ch": int(args.n_ch),
                    "fs_hz": float(fs_hz),
                    "seq": args.seq,
                    "backend": args.backend,
                    "ack_strategy": ack_state.get("strategy", "none"),
                }
            )

        time.sleep(0.2)
        total_ms = (time.perf_counter() - t_start) * 1000.0
        submit_vals = [r["submit_ms"] for r in rows]
        stats = summarize_values(submit_vals)
        summary = {
            "bench": "A2",
            "tool": "fastplotlib",
            "status": "OK",
            "count_ok": int(len(rows)),
            "count_fail": 0,
            "p50_submit_ms": stats["p50"],
            "p95_submit_ms": stats["p95"],
            "total_ms": float(total_ms),
            "ack_strategy": ack_state.get("strategy", "none"),
            "backend": backend_info.get("backend", args.backend),
            "canvas_class": backend_info.get("canvas_class"),
            "canvas_vsync": backend_info.get("vsync"),
            "canvas_max_fps": backend_info.get("max_fps"),
            "meta": meta,
        }
        fieldnames = [
            "step_idx",
            "op",
            "world_dx",
            "zoom_factor",
            "submit_ms",
            "ok",
            "timeout",
            "tool",
            "bench",
            "window_s",
            "n_ch",
            "fs_hz",
            "seq",
            "backend",
            "ack_strategy",
        ]
        write_steps_csv(outdir / "steps.csv", rows, fieldnames)
        write_summary(outdir, summary)
        bench_json_line(summary)
    except Exception as exc:
        status = 1
        summary = {"bench": "A2", "tool": "fastplotlib", "status": "ERROR", "error": str(exc)}
        write_summary(outdir, summary)
        bench_json_line(summary)
    finally:
        if fig is not None:
            safe_close(fig)

    return status


if __name__ == "__main__":
    raise SystemExit(main())
