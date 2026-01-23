from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from benchkit.bench_defaults import PAN_STEP_FRACTION, ZOOM_IN_FACTOR, ZOOM_OUT_FACTOR
from benchkit.lexicon import SEQ_PAN, SEQ_PAN_ZOOM, SEQ_ZOOM_IN, SEQ_ZOOM_OUT


@dataclass(frozen=True)
class StepSpec:
    index: int
    label: str
    x0: float
    x1: float
    y0: float
    y1: float


def _clamp_range(x0: float, x1: float, lo: float, hi: float) -> Tuple[float, float]:
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

        x0, x1 = _clamp_range(x0, x1, lo, hi)
        rng.append((x0, x1))
    return rng


def _step_label(sequence: str, idx: int) -> str:
    if sequence == SEQ_PAN:
        return "PAN"
    if sequence == SEQ_ZOOM_IN:
        return "ZOOM_IN"
    if sequence == SEQ_ZOOM_OUT:
        return "ZOOM_OUT"
    if sequence == SEQ_PAN_ZOOM:
        return "PAN" if idx % 2 == 0 else "ZOOM_IN"
    return str(sequence)


def build_step_specs(
    sequence: str,
    lo: float,
    hi: float,
    window_s: float,
    steps: int,
    y0: float,
    y1: float,
) -> List[StepSpec]:
    ranges = build_ranges(sequence, lo, hi, window_s, steps)
    specs: List[StepSpec] = []
    for idx, (x0, x1) in enumerate(ranges):
        specs.append(
            StepSpec(
                index=int(idx),
                label=_step_label(sequence, idx),
                x0=float(x0),
                x1=float(x1),
                y0=float(y0),
                y1=float(y1),
            )
        )
    return specs


def build_step_rows(
    step_specs: List[StepSpec],
    issue_times: Iterable[Optional[float]],
    ack_times: Iterable[Optional[float]],
    statuses: Iterable[str],
) -> List[Dict[str, Any]]:
    issue_list = list(issue_times)
    ack_list = list(ack_times)
    status_list = list(statuses)
    rows: List[Dict[str, Any]] = []
    for spec in step_specs:
        t_issue = issue_list[spec.index] if spec.index < len(issue_list) else None
        t_ack = ack_list[spec.index] if spec.index < len(ack_list) else None
        latency_s = (t_ack - t_issue) if (t_issue is not None and t_ack is not None) else None
        status = status_list[spec.index] if spec.index < len(status_list) else "SKIP"
        rows.append(
            {
                "step_index": int(spec.index),
                "step_label": spec.label,
                "t_issue": t_issue,
                "t_ack": t_ack,
                "latency_s": latency_s,
                "xlim0": spec.x0,
                "xlim1": spec.x1,
                "ylim0": spec.y0,
                "ylim1": spec.y1,
                "status": status,
            }
        )
    return rows


def summarize_latency_s(latencies_s: Iterable[float]) -> Dict[str, Any]:
    a = np.asarray(list(latencies_s), dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {
            "latency_p50_s": float("nan"),
            "latency_p95_s": float("nan"),
            "latency_max_s": float("nan"),
            "count": 0,
        }
    return {
        "latency_p50_s": float(np.percentile(a, 50)),
        "latency_p95_s": float(np.percentile(a, 95)),
        "latency_max_s": float(np.max(a)),
        "count": int(a.size),
    }
