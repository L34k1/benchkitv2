#!/usr/bin/env python3
"""scripts/desktop/bench_fastplotlib.py

fastplotlib benchmark entrypoint (EDF/NWB).
Implements TFFR/A1/A2 using axis range updates when available.
"""

from __future__ import annotations

import sys

import argparse
import importlib.util
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchkit.common import out_dir
from benchkit.bench_defaults import (
    DEFAULT_STEPS,
    DEFAULT_TARGET_INTERVAL_MS,
    DEFAULT_WINDOW_S,
    PAN_STEP_FRACTION,
    ZOOM_IN_FACTOR,
    ZOOM_OUT_FACTOR,
    default_load_duration_s,
)
from benchkit.lexicon import (
    BENCH_A1,
    BENCH_A2,
    BENCH_TFFR,
    FMT_EDF,
    FMT_NWB,
    SEQ_PAN,
    SEQ_PAN_ZOOM,
    SEQ_ZOOM_IN,
    SEQ_ZOOM_OUT,
    TOOL_FASTPLOTLIB,
)
from benchkit.loaders import load_edf_segment_pyedflib, load_nwb_segment_pynwb
from benchkit.output_contract import (
    steps_from_latencies,
    write_manifest_contract,
    write_steps_csv,
    write_steps_summary,
    write_tffr_csv,
    write_tffr_summary,
)


def normalize_bench_id(bench_id: str) -> str:
    if bench_id == "A1":
        return BENCH_A1
    if bench_id == "A2":
        return BENCH_A2
    return bench_id


def clamp_range(x0: float, x1: float, lo: float, hi: float) -> Tuple[float, float]:
    w = x1 - x0
    if w <= 0:
        w = 1e-6
    if x0 < lo:
        x0, x1 = lo, lo + w
    if x1 > hi:
        x1, x0 = hi, hi - w
    if x0 < lo:
        x0, x1 = lo, hi
    return x0, x1


def build_ranges(sequence: str, lo: float, hi: float, window_s: float, steps: int) -> List[Tuple[float, float]]:
    x0, x1 = lo, min(lo + window_s, hi)
    w = x1 - x0
    pan_step = w * PAN_STEP_FRACTION
    rng: List[Tuple[float, float]] = []

    for i in range(steps):
        if sequence == SEQ_PAN:
            x0, x1 = x0 + pan_step, x1 + pan_step
            if x1 > hi or x0 < lo:
                pan_step *= -1.0
        elif sequence == SEQ_ZOOM_IN:
            cx = 0.5 * (x0 + x1)
            w = max(w * ZOOM_IN_FACTOR, window_s * 0.10)
            x0, x1 = cx - 0.5 * w, cx + 0.5 * w
        elif sequence == SEQ_ZOOM_OUT:
            cx = 0.5 * (x0 + x1)
            w = min(w * ZOOM_OUT_FACTOR, hi - lo)
            x0, x1 = cx - 0.5 * w, cx + 0.5 * w
        elif sequence == SEQ_PAN_ZOOM:
            if i % 2 == 0:
                x0, x1 = x0 + pan_step, x1 + pan_step
                if x1 > hi or x0 < lo:
                    pan_step *= -1.0
            else:
                cx = 0.5 * (x0 + x1)
                w = max(min(w * ZOOM_IN_FACTOR, hi - lo), window_s * 0.10)
                x0, x1 = cx - 0.5 * w, cx + 0.5 * w
        else:
            raise ValueError(sequence)

        x0, x1 = clamp_range(x0, x1, lo, hi)
        rng.append((x0, x1))
    return rng


def _ensure_qt_output_context() -> Tuple[bool, str]:
    try:
        import fastplotlib.layouts._frame._frame as frame
    except Exception as exc:
        return False, f"fastplotlib frame import failed: {exc}"
    if hasattr(frame, "QOutputContext"):
        return True, ""
    try:
        from PyQt5 import QtWidgets
    except Exception as exc:
        return False, f"PyQt5 unavailable: {exc}"

    class QOutputContext(QtWidgets.QWidget):
        def __init__(self, frame=None, make_toolbar=False, add_widgets=None, **_):
            super().__init__(parent=None)
            self.frame = frame
            self.toolbar = None
            layout = QtWidgets.QVBoxLayout(self)
            if add_widgets is None:
                add_widgets = []
            layout.addWidget(self.frame.canvas)
            for w in add_widgets:
                try:
                    w.setParent(self)
                except Exception:
                    pass
                layout.addWidget(w)
            self.setLayout(layout)
            try:
                self.resize(*self.frame._starting_size)
            except Exception:
                pass
            self.show()

        def close(self):
            try:
                self.frame.canvas.close()
            except Exception:
                pass
            super().close()

    frame.QOutputContext = QOutputContext
    return True, ""


def _create_plot(fpl, canvas_backend: str | None) -> Tuple[object, object, List[str]]:
    errors: List[str] = []
    fig = None
    ax = None

    if hasattr(fpl, "Figure"):
        try:
            fig = fpl.Figure()
            if hasattr(fig, "__getitem__"):
                ax = fig[0, 0]
            elif hasattr(fig, "plot"):
                ax = fig.plot
            elif hasattr(fig, "axes"):
                axes = fig.axes
                if isinstance(axes, (list, tuple)) and axes:
                    ax = axes[0]
        except Exception as exc:
            errors.append(f"Figure(): {exc}")

    if ax is None and hasattr(fpl, "figure"):
        try:
            fig = fpl.figure()
            if hasattr(fig, "__getitem__"):
                ax = fig[0, 0]
            elif hasattr(fig, "plot"):
                ax = fig.plot
        except Exception as exc:
            errors.append(f"figure(): {exc}")

    if ax is None and hasattr(fpl, "Plot"):
        try:
            if canvas_backend:
                ax = fpl.Plot(canvas=canvas_backend)
            else:
                ax = fpl.Plot()
            fig = getattr(ax, "figure", None) or ax
        except Exception as exc:
            errors.append(f"Plot(): {exc}")

    if ax is None and hasattr(fpl, "PlotWidget"):
        try:
            ax = fpl.PlotWidget()
            fig = getattr(ax, "figure", None) or ax
        except Exception as exc:
            errors.append(f"PlotWidget(): {exc}")

    return fig, ax, errors


def _set_xrange(ax: object, x0: float, x1: float) -> bool:
    if hasattr(ax, "set_xlim"):
        ax.set_xlim((x0, x1))
        return True
    if hasattr(ax, "set_range"):
        ax.set_range(x=(x0, x1))
        return True
    if hasattr(ax, "xlim"):
        try:
            ax.xlim = (x0, x1)
            return True
        except Exception:
            return False
    cam = getattr(ax, "camera", None)
    if cam is not None and hasattr(cam, "get_state") and hasattr(cam, "set_state"):
        try:
            state = cam.get_state()
            pos = np.asarray(state.get("position", [0.0, 0.0, 0.0]), dtype=float)
            pos[0] = 0.5 * (x0 + x1)
            state["position"] = pos
            state["width"] = float(max(1e-9, x1 - x0))
            cam.set_state(state)
            return True
        except Exception:
            return False
    return False


def _request_draw(fig: object, ax: object) -> bool:
    canvas = getattr(fig, "canvas", None)
    if canvas is None and hasattr(ax, "canvas"):
        canvas = ax.canvas
    if canvas is not None and hasattr(canvas, "request_draw"):
        canvas.request_draw()
        return True
    if canvas is not None and hasattr(canvas, "draw"):
        canvas.draw()
        return True
    if hasattr(fig, "request_draw"):
        fig.request_draw()
        return True
    if hasattr(ax, "request_draw"):
        ax.request_draw()
        return True
    return False


def _show(fig: object, ax: object) -> bool:
    if hasattr(fig, "show"):
        fig.show()
        return True
    if hasattr(ax, "show"):
        ax.show()
        return True
    return False


def _write_skip(out: Path, args: argparse.Namespace, reason: str, detail: str) -> None:
    if args.bench_id == BENCH_TFFR:
        write_tffr_csv(out, run_id=0, tffr_ms=float("nan"))
        write_tffr_summary(
            out,
            bench_id=BENCH_TFFR,
            tool_id=TOOL_FASTPLOTLIB,
            fmt=args.format,
            window_s=float(args.window_s),
            n_channels=int(args.n_ch),
            tffr_ms=float("nan"),
            extra={"status": reason, "skip_reason": detail},
        )
    else:
        steps_rows: List[dict] = []
        write_steps_csv(out, steps_rows)
        write_steps_summary(
            out,
            steps_rows,
            extra={
                "bench_id": args.bench_id,
                "tool_id": TOOL_FASTPLOTLIB,
                "sequence": args.sequence,
                "steps": int(args.steps),
                "target_interval_ms": float(args.target_interval_ms),
                "format": args.format,
                "status": reason,
                "skip_reason": detail,
            },
        )


def load_segment(args: argparse.Namespace) -> Tuple[List[float], List[List[float]]]:
    if args.format == FMT_EDF:
        seg = load_edf_segment_pyedflib(args.file, args.load_start_s, args.load_duration_s, args.n_ch)
    else:
        seg = load_nwb_segment_pynwb(
            args.file,
            args.load_start_s,
            args.load_duration_s,
            args.n_ch,
            series_path=args.nwb_series_path,
            time_dim=args.nwb_time_dim,
        )
    t = seg.times_s.astype(float).tolist()
    data = seg.data.astype(float).tolist()
    return t, data


def main() -> None:
    ap = argparse.ArgumentParser(description="fastplotlib benchmark.")
    ap.add_argument("--bench-id", choices=[BENCH_TFFR, BENCH_A1, BENCH_A2, "A1", "A2"], required=True)
    ap.add_argument("--format", choices=[FMT_EDF, FMT_NWB], required=True)
    ap.add_argument("--file", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, default=Path("outputs"))
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--sequence", choices=[SEQ_PAN, SEQ_ZOOM_IN, SEQ_ZOOM_OUT, SEQ_PAN_ZOOM], default=SEQ_PAN)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S)
    ap.add_argument("--target-interval-ms", type=float, default=DEFAULT_TARGET_INTERVAL_MS)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--n-ch", type=int, default=16)
    ap.add_argument("--load-start-s", type=float, default=0.0)
    ap.add_argument("--load-duration-s", type=float, default=None)
    ap.add_argument("--nwb-series-path", type=str, default=None)
    ap.add_argument("--nwb-time-dim", type=str, default="auto", choices=["auto", "time_first", "time_last"])
    ap.add_argument("--canvas-backend", choices=["auto", "qt", "glfw"], default="auto")
    args = ap.parse_args()
    args.bench_id = normalize_bench_id(args.bench_id)
    if args.load_duration_s is None:
        args.load_duration_s = default_load_duration_s(args.window_s)

    if args.runs != 1:
        print(f"[WARN] forcing runs=1 (requested {args.runs})")
        args.runs = 1
    canvas_backend = None
    if args.canvas_backend == "glfw":
        canvas_backend = "glfw"
    elif args.canvas_backend == "qt":
        canvas_backend = None
    else:
        try:
            import glfw  # noqa: F401

            canvas_backend = "glfw"
        except Exception:
            canvas_backend = None
    out = out_dir(args.out_root, args.bench_id, TOOL_FASTPLOTLIB, args.tag)
    write_manifest_contract(
        out,
        bench_id=args.bench_id,
        tool_id=TOOL_FASTPLOTLIB,
        fmt=args.format,
        file_path=args.file,
        window_s=float(args.window_s),
        n_channels=int(args.n_ch),
        sequence=args.sequence,
        overlay=None,
        run_id=0,
        steps_target=int(args.steps),
        extra={"format": args.format, "canvas_backend": canvas_backend},
    )

    if importlib.util.find_spec("fastplotlib") is None:
        print("SKIP_UNSUPPORTED_DEP: fastplotlib is not installed.")
        _write_skip(out, args, "SKIP_UNSUPPORTED_DEP", "fastplotlib is not installed")
        return
    try:
        import PyQt5  # noqa: F401
    except Exception:
        pass
    import fastplotlib as fpl
    if canvas_backend is None:
        ok_qt, qt_detail = _ensure_qt_output_context()
        if not ok_qt:
            print(f"SKIP_UNSUPPORTED_DEP: {qt_detail}")
            _write_skip(out, args, "SKIP_UNSUPPORTED_DEP", qt_detail)
            return

    t, data = load_segment(args)
    t_arr = np.asarray(t, dtype=float)
    data_arr = np.asarray(data, dtype=float)
    fig, ax, errors = _create_plot(fpl, canvas_backend)
    if fig is None or ax is None or not hasattr(ax, "add_line"):
        detail = errors[0] if errors else "fastplotlib figure/plot API not found"
        print(f"SKIP_UNSUPPORTED_API: {detail}")
        _write_skip(out, args, "SKIP_UNSUPPORTED_API", detail)
        return
    try:
        for ch in range(min(len(data_arr), args.n_ch)):
            xy = np.column_stack([t_arr, data_arr[ch]])
            ax.add_line(xy)
    except Exception as exc:
        detail = f"add_line failed: {exc}"
        print(f"SKIP_UNSUPPORTED_API: {detail}")
        _write_skip(out, args, "SKIP_UNSUPPORTED_API", detail)
        return
    t0 = time.perf_counter()
    try:
        shown = _show(fig, ax)
    except Exception as exc:
        detail = f"show failed: {exc}"
        print(f"SKIP_UNSUPPORTED_API: {detail}")
        _write_skip(out, args, "SKIP_UNSUPPORTED_API", detail)
        return
    if not shown:
        detail = "fastplotlib figure/plot has no show()"
        print(f"SKIP_UNSUPPORTED_API: {detail}")
        _write_skip(out, args, "SKIP_UNSUPPORTED_API", detail)
        return
    tffr_s = time.perf_counter() - t0

    if args.bench_id == BENCH_TFFR:
        tffr_ms = float(tffr_s) * 1000.0
        write_tffr_csv(out, run_id=0, tffr_ms=tffr_ms)
        write_tffr_summary(
            out,
            bench_id=BENCH_TFFR,
            tool_id=TOOL_FASTPLOTLIB,
            fmt=args.format,
            window_s=float(args.window_s),
            n_channels=int(args.n_ch),
            tffr_ms=tffr_ms,
        )
        return

    lo, hi = float(t[0]), float(t[-1])
    ranges = build_ranges(args.sequence, lo, hi, float(args.window_s), int(args.steps))
    lat_ms: List[float] = []
    interval_s = float(args.target_interval_ms) / 1000.0
    start = time.perf_counter()
    cam = getattr(ax, "camera", None)
    cam_ok = cam is not None and hasattr(cam, "get_state") and hasattr(cam, "set_state")
    if not (hasattr(ax, "set_xlim") or hasattr(ax, "set_range") or hasattr(ax, "xlim") or cam_ok):
        detail = "fastplotlib plot lacks x-range setter"
        print(f"SKIP_UNSUPPORTED_API: {detail}")
        _write_skip(out, args, "SKIP_UNSUPPORTED_API", detail)
        return
    if not _request_draw(fig, ax):
        detail = "fastplotlib canvas does not support request_draw/draw"
        print(f"SKIP_UNSUPPORTED_API: {detail}")
        _write_skip(out, args, "SKIP_UNSUPPORTED_API", detail)
        return
    for idx, (x0, x1) in enumerate(ranges):
        if args.bench_id == BENCH_A2:
            target = start + idx * interval_s
            while time.perf_counter() < target:
                pass
        t_issue = time.perf_counter()
        if not _set_xrange(ax, x0, x1):
            detail = "fastplotlib plot does not support set_xlim/set_range"
            print(f"SKIP_UNSUPPORTED_API: {detail}")
            _write_skip(out, args, "SKIP_UNSUPPORTED_API", detail)
            return
        _request_draw(fig, ax)
        t_paint = time.perf_counter()
        lat_ms.append((t_paint - t_issue) * 1000.0)

    steps_rows = steps_from_latencies(lat_ms, steps_target=int(args.steps))
    write_steps_csv(out, steps_rows)
    write_steps_summary(
        out,
        steps_rows,
        extra={
            "bench_id": args.bench_id,
            "tool_id": TOOL_FASTPLOTLIB,
            "sequence": args.sequence,
            "steps": int(args.steps),
            "target_interval_ms": float(args.target_interval_ms),
            "latency_ms_mean": float(sum(lat_ms) / max(1, len(lat_ms))),
        },
    )


if __name__ == "__main__":
    main()
