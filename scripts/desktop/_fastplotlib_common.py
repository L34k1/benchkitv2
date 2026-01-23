from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchkit.bench_defaults import default_load_duration_s
from benchkit.common import ensure_dir, utc_stamp, write_json
from benchkit.loaders import decimate_for_display, load_edf_segment_pyedflib, load_nwb_segment_pynwb


@dataclass(frozen=True)
class Step:
    op: str
    world_dx: float
    zoom_factor: Optional[float]
    x_min: float
    x_max: float
    x_span: float


def _package_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _git_hash() -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        return None
    return None


def _python_info() -> str:
    return sys.version.replace("\n", " ")


def _env_info() -> Dict[str, Any]:
    return {
        "timestamp_utc": utc_stamp(),
        "python": _python_info(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cwd": os.getcwd(),
    }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


def _load_npz(path: Path, fs_hz: float, n_ch: int) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
    arr = np.load(path, allow_pickle=False)
    if "data" not in arr:
        raise ValueError("NPZ must contain 'data' array")
    data = np.asarray(arr["data"], dtype=np.float32)
    if data.ndim != 2:
        raise ValueError("NPZ 'data' must be 2D (n_channels, n_samples)")
    if data.shape[0] < n_ch:
        raise ValueError(f"NPZ has {data.shape[0]} channels, requested {n_ch}")
    data = data[:n_ch]
    if "times" in arr:
        times = np.asarray(arr["times"], dtype=np.float32)
    else:
        if fs_hz <= 0:
            raise ValueError("Provide --fs-hz when NPZ has no 'times' array")
        times = np.arange(data.shape[1], dtype=np.float32) / float(fs_hz)
    if fs_hz <= 0:
        fs_hz = float(arr["fs_hz"]) if "fs_hz" in arr else float("nan")
    meta = {
        "path": str(path),
        "n_ch_total": int(data.shape[0]),
        "n_ch_used": int(n_ch),
        "effective_n_ch": int(n_ch),
        "duration_s": float(times[-1] - times[0]) if times.size > 1 else float("nan"),
    }
    return times, data, float(fs_hz), meta


def load_data(
    path: Path,
    fs_hz: float,
    window_s: float,
    n_ch: int,
    *,
    max_points_per_trace: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        times, data, fs, meta = _load_npz(path, fs_hz, n_ch)
    elif suffix == ".edf":
        duration_s = default_load_duration_s(window_s)
        seg = load_edf_segment_pyedflib(path, 0.0, duration_s, n_ch)
        times, data, dec = decimate_for_display(seg.times_s, seg.data, max_points_per_trace)
        meta = dict(seg.meta)
        meta["decim_factor"] = int(dec)
        meta["n_points_per_trace"] = int(times.shape[0])
        fs = float(seg.fs_hz)
    elif suffix == ".nwb":
        duration_s = default_load_duration_s(window_s)
        seg = load_nwb_segment_pynwb(path, 0.0, duration_s, n_ch)
        times, data, dec = decimate_for_display(seg.times_s, seg.data, max_points_per_trace)
        meta = dict(seg.meta)
        meta["decim_factor"] = int(dec)
        meta["n_points_per_trace"] = int(times.shape[0])
        fs = float(seg.fs_hz)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
    return times.astype(np.float32), data.astype(np.float32), float(fs), meta


def build_figure(
    data: np.ndarray,
    times: np.ndarray,
    fs_hz: float,
    initial_view: Dict[str, Any],
    backend_opts: Dict[str, Any],
) -> Tuple[Any, Any, Dict[str, Any], Dict[str, Any]]:
    import fastplotlib as fpl

    backend = backend_opts.get("backend")
    canvas_kwargs = backend_opts.get("canvas_kwargs")
    fig_kwargs: Dict[str, Any] = {"size": backend_opts.get("size", (900, 600))}
    if backend and backend != "auto":
        fig_kwargs["canvas"] = backend
    if canvas_kwargs:
        fig_kwargs["canvas_kwargs"] = canvas_kwargs

    fig = fpl.Figure(**fig_kwargs)
    subplot = fig[0, 0]

    gain = float(initial_view["gain"])
    spacing = float(initial_view["spacing"])
    offsets = np.arange(data.shape[0], dtype=np.float32) * spacing
    for ch in range(data.shape[0]):
        y = data[ch] * gain + offsets[ch]
        xy = np.column_stack([times, y]).astype(np.float32, copy=False)
        subplot.add_line(xy, thickness=1.0, colors="w", uniform_color=True)

    cam = subplot.camera
    try:
        state = cam.get_state()
    except Exception:
        state = {}
    x_min = float(initial_view["x_min"])
    x_max = float(initial_view["x_max"])
    y_min = float(initial_view["y_min"])
    y_max = float(initial_view["y_max"])
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    width = max(1e-9, x_max - x_min)
    height = max(1e-9, y_max - y_min)
    new_state = dict(state)
    new_state.update(
        {
            "position": np.array([x_mid, y_mid, 0.0], dtype=float),
            "width": float(width),
            "height": float(height),
            "maintain_aspect": False,
            "fov": 0.0,
        }
    )
    try:
        cam.set_state(new_state)
    except Exception:
        pass

    backend_info: Dict[str, Any] = {
        "backend": backend,
        "canvas_class": None,
        "vsync": None,
        "max_fps": None,
    }
    canvas = getattr(fig, "canvas", None)
    if canvas is not None:
        backend_info["canvas_class"] = canvas.__class__.__name__
        if "canvas_kwargs" in fig_kwargs:
            if hasattr(canvas, "vsync"):
                try:
                    canvas.vsync = bool(canvas_kwargs.get("vsync", True))
                except Exception:
                    pass
            if hasattr(canvas, "max_fps") and int(canvas_kwargs.get("max_fps", 0)) > 0:
                try:
                    canvas.max_fps = int(canvas_kwargs.get("max_fps", 0))
                except Exception:
                    pass
        backend_info["vsync"] = getattr(canvas, "vsync", None)
        backend_info["max_fps"] = getattr(canvas, "max_fps", None)

    return fig, subplot, new_state, backend_info


def make_sequence(
    seq_name: str,
    n_steps: int,
    seed: int,
    initial_view: Dict[str, Any],
    params: Dict[str, Any],
) -> List[Step]:
    del seed
    x_min = float(initial_view["x_min"])
    x_max = float(initial_view["x_max"])
    steps: List[Step] = []
    zoom_in = float(params.get("zoom_in", 1.25))
    zoom_out = float(params.get("zoom_out", 0.8))

    for idx in range(int(n_steps)):
        span = max(1e-9, x_max - x_min)
        if seq_name == "PAN":
            world_dx = 0.1 * span
            x_min += world_dx
            x_max += world_dx
            steps.append(Step("PAN", world_dx, None, x_min, x_max, span))
        elif seq_name == "ZOOM_IN":
            factor = zoom_in
            new_span = span / factor
            cx = 0.5 * (x_min + x_max)
            x_min = cx - 0.5 * new_span
            x_max = cx + 0.5 * new_span
            steps.append(Step("ZOOM_IN", 0.0, factor, x_min, x_max, span))
        elif seq_name == "ZOOM_OUT":
            factor = zoom_out
            new_span = span / factor
            cx = 0.5 * (x_min + x_max)
            x_min = cx - 0.5 * new_span
            x_max = cx + 0.5 * new_span
            steps.append(Step("ZOOM_OUT", 0.0, factor, x_min, x_max, span))
        elif seq_name == "PAN_ZOOM":
            if idx % 2 == 0:
                world_dx = 0.1 * span
                x_min += world_dx
                x_max += world_dx
                steps.append(Step("PAN", world_dx, None, x_min, x_max, span))
            else:
                factor = zoom_in
                new_span = span / factor
                cx = 0.5 * (x_min + x_max)
                x_min = cx - 0.5 * new_span
                x_max = cx + 0.5 * new_span
                steps.append(Step("ZOOM_IN", 0.0, factor, x_min, x_max, span))
        else:
            raise ValueError(f"Unknown sequence: {seq_name}")
    return steps


def compute_screen_delta_for_world_shift(subplot: Any, world_dx: float) -> Tuple[float, float]:
    rect = subplot.viewport.rect
    if hasattr(subplot, "map_world_to_screen"):
        try:
            center = np.array([0.0, 0.0, 0.0], dtype=float)
            p0 = subplot.map_world_to_screen(center)
            p1 = subplot.map_world_to_screen(center + np.array([world_dx, 0.0, 0.0], dtype=float))
            if p0 is not None and p1 is not None:
                return float(p1[0] - p0[0]), 0.0
        except Exception:
            pass
    if hasattr(subplot, "map_screen_to_world"):
        try:
            cx = rect[0] + rect[2] * 0.5
            cy = rect[1] + rect[3] * 0.5
            w0 = subplot.map_screen_to_world((cx, cy))
            w1 = subplot.map_screen_to_world((cx + 1.0, cy))
            if w0 is not None and w1 is not None:
                world_per_px = float(w1[0] - w0[0])
                if world_per_px != 0:
                    return float(world_dx / world_per_px), 0.0
        except Exception:
            pass
    cam = getattr(subplot, "camera", None)
    span = None
    if cam is not None and hasattr(cam, "get_state"):
        try:
            span = float(cam.get_state().get("width"))
        except Exception:
            span = None
    if span is None or span <= 0:
        span = abs(world_dx) if world_dx else 1.0
    width_px = rect[2] if rect is not None else 1.0
    dx_px = (float(world_dx) / float(span)) * float(width_px)
    return dx_px, 0.0


def apply_step(subplot: Any, step: Step) -> None:
    controller = subplot.controller
    rect = subplot.viewport.rect
    if step.op == "PAN":
        dx_px, dy_px = compute_screen_delta_for_world_shift(subplot, step.world_dx)
        controller.pan((dx_px, dy_px), rect=rect, animate=False)
    elif step.op in ("ZOOM_IN", "ZOOM_OUT"):
        factor = float(step.zoom_factor or 1.0)
        delta = math.log2(factor)
        if hasattr(controller, "zoom_to_point"):
            cx = rect[0] + rect[2] * 0.5
            cy = rect[1] + rect[3] * 0.5
            controller.zoom_to_point(delta, pos=(cx, cy), rect=rect, animate=False)
        else:
            controller.zoom((delta, delta), rect=rect, animate=False)
    else:
        raise ValueError(step.op)


def _get_canvas(fig: Any) -> Any:
    canvas = getattr(fig, "canvas", None)
    if canvas is not None:
        return canvas
    return None


def _get_renderer(fig: Any) -> Any:
    return getattr(fig, "renderer", None)


def _pick_ack_strategy(fig: Any) -> List[str]:
    canvas = _get_canvas(fig)
    strategies: List[str] = []
    if canvas is not None and hasattr(canvas, "add_event_handler"):
        strategies.append("animate")
    if canvas is not None and hasattr(canvas, "force_draw"):
        strategies.append("force_draw")
    renderer = _get_renderer(fig)
    if renderer is not None and hasattr(renderer, "request_draw") and hasattr(renderer, "snapshot"):
        strategies.append("snapshot")
    if not strategies:
        strategies.append("snapshot")
    return strategies


def ack_frame(fig: Any, timeout_s: float, strategy: str, state: Dict[str, Any]) -> bool:
    if strategy == "auto":
        preferred = _pick_ack_strategy(fig)
        # If we already found a working strategy, reuse it.
        if state.get("strategy") in preferred:
            strategy = state["strategy"]
        else:
            for cand in preferred:
                ok = ack_frame(fig, timeout_s, cand, state)
                if ok:
                    state["strategy"] = cand
                    return True
            return False
    state["strategy"] = strategy
    canvas = _get_canvas(fig)
    renderer = _get_renderer(fig)

    if strategy == "animate" and canvas is not None and hasattr(canvas, "add_event_handler"):
        if "tick" not in state:
            state["tick"] = 0
            state["tick_at"] = 0.0

            def _on_animate(_ev) -> None:
                state["tick"] += 1
                state["tick_at"] = time.perf_counter()

            canvas.add_event_handler(_on_animate, "animate")
            state["handler"] = _on_animate

        start_tick = state.get("tick", 0)
        start_time = time.perf_counter()
        while (time.perf_counter() - start_time) < timeout_s:
            if state.get("tick", 0) > start_tick:
                return True
            time.sleep(0.001)
        return False

    if strategy == "force_draw" and canvas is not None and hasattr(canvas, "force_draw"):
        try:
            canvas.force_draw()
            return True
        except Exception:
            return False

    if strategy == "snapshot" and renderer is not None and hasattr(renderer, "snapshot"):
        if hasattr(renderer, "request_draw"):
            try:
                renderer.request_draw()
            except Exception:
                pass
        start_time = time.perf_counter()
        last_hash = state.get("snapshot_hash")
        while (time.perf_counter() - start_time) < timeout_s:
            try:
                img = renderer.snapshot()
                h = hash(img.tobytes())
                if last_hash is None or h != last_hash:
                    state["snapshot_hash"] = h
                    return True
            except Exception:
                pass
            time.sleep(0.01)
        return False

    return False


def request_draw(fig: Any) -> None:
    canvas = _get_canvas(fig)
    renderer = _get_renderer(fig)
    if canvas is not None and hasattr(canvas, "request_draw"):
        try:
            canvas.request_draw()
            return
        except Exception:
            pass
    if renderer is not None and hasattr(renderer, "request_draw"):
        try:
            renderer.request_draw()
            return
        except Exception:
            pass


def safe_close(fig: Any) -> None:
    canvas = _get_canvas(fig)
    if canvas is not None and hasattr(canvas, "close"):
        try:
            canvas.close()
        except Exception:
            pass
    if hasattr(fig, "close"):
        try:
            fig.close()
        except Exception:
            pass


def show_figure(fig: Any) -> bool:
    if hasattr(fig, "show"):
        try:
            fig.show()
            return True
        except Exception:
            return False
    canvas = _get_canvas(fig)
    if canvas is not None and hasattr(canvas, "show"):
        try:
            canvas.show()
            return True
        except Exception:
            return False
    return False


def write_manifest(out_dir: Path, args: Dict[str, Any], initial_view: Dict[str, Any], camera_state: Dict[str, Any]) -> None:
    ensure_dir(out_dir)
    manifest = {
        "timestamp_utc": utc_stamp(),
        "args": args,
        "initial_view": initial_view,
        "camera_state": camera_state,
        "env": _env_info(),
        "git_hash": _git_hash(),
        "packages": {
            "fastplotlib": _package_version("fastplotlib"),
            "pygfx": _package_version("pygfx"),
            "rendercanvas": _package_version("rendercanvas"),
            "wgpu": _package_version("wgpu"),
        },
    }
    write_json(Path(out_dir) / "manifest.json", _jsonable(manifest))


def write_steps_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    ensure_dir(path.parent)
    import csv

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_summary(out_dir: Path, summary: Dict[str, Any]) -> None:
    write_json(Path(out_dir) / "summary.json", _jsonable(summary))


def summarize_values(values: Iterable[float]) -> Dict[str, Any]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"p50": float("nan"), "p95": float("nan")}
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def bench_json_line(summary: Dict[str, Any]) -> None:
    print(f"BENCH_JSON: {json.dumps(_jsonable(summary), ensure_ascii=True)}")
