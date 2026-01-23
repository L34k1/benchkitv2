"""Quick benchmarks that generate CSV outputs per tech."""

from __future__ import annotations

import csv
import importlib
import importlib.util
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from benchkitv2.web_console import collect_html_console


@dataclass(frozen=True)
class Tech:
    tool_id: str
    modules: tuple[str, ...]
    kind: str


TECHS: tuple[Tech, ...] = (
    Tech(tool_id="pyqtgraph", modules=("pyqtgraph",), kind="desktop"),
    Tech(tool_id="vispy", modules=("vispy",), kind="desktop"),
    Tech(tool_id="mne", modules=("mne",), kind="desktop"),
    Tech(tool_id="fastplotlib", modules=("fastplotlib",), kind="desktop"),
    Tech(tool_id="plotly", modules=("plotly",), kind="web"),
    Tech(tool_id="d3", modules=(), kind="web"),
    Tech(tool_id="pyedflib", modules=("pyedflib",), kind="io"),
    Tech(tool_id="neo", modules=("neo",), kind="io"),
    Tech(tool_id="pynwb", modules=("pynwb",), kind="io"),
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _techs_by_name(names: Iterable[str]) -> list[Tech]:
    if list(names) == ["all"]:
        return list(TECHS)
    lookup = {t.tool_id: t for t in TECHS}
    missing = [name for name in names if name not in lookup]
    if missing:
        raise SystemExit(f"Unknown tech(s): {', '.join(missing)}")
    return [lookup[name] for name in names]


def _modules_available(modules: tuple[str, ...]) -> tuple[bool, str]:
    if not modules:
        return True, "no-python-module"
    for module in modules:
        if importlib.util.find_spec(module) is None:
            return False, f"missing {module}"
    return True, "available"


def _write_csv_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = list(row.keys())
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _safe_value(value: object) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def _benchmark_import(tech: Tech) -> dict[str, object]:
    start = time.perf_counter()
    status = "ok"
    detail = "imported"
    try:
        for module in tech.modules:
            importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - surface exact error
        status = "fail"
        detail = f"{type(exc).__name__}: {exc}"
    end = time.perf_counter()
    return {
        "timestamp_start": _utc_stamp(),
        "timestamp_end": _utc_stamp(),
        "duration_s": f"{end - start:.6f}",
        "bench_id": "import",
        "tool_id": tech.tool_id,
        "status": status,
        "detail": _safe_value(detail),
    }


def _generate_web_smoke_html(out_dir: Path, tech: Tech) -> Path:
    _ensure_dir(out_dir)
    html_path = out_dir / f"{tech.tool_id}_bench.html"
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>benchkitv2 web smoke</title>
</head>
<body>
  <script>
    const start = performance.now();
    const payload = {
      bench_id: "web_smoke",
      tool_id: "%s",
      timestamp_start: new Date().toISOString(),
      duration_ms: 0,
      status: "ok"
    };
    requestAnimationFrame(() => {
      const end = performance.now();
      payload.duration_ms = Math.round(end - start);
      payload.timestamp_end = new Date().toISOString();
      console.log("BENCH_JSON:" + JSON.stringify(payload));
    });
  </script>
</body>
</html>
""" % (
        tech.tool_id
    )
    html_path.write_text(html, encoding="utf-8")
    return html_path


def _find_data_file(data_dir: Path) -> tuple[Path, str] | None:
    if not data_dir.exists():
        return None
    for ext, fmt in ((".edf", "EDF"), (".nwb", "NWB")):
        matches = sorted(data_dir.rglob(f"*{ext}"))
        if matches:
            return matches[0], fmt
    return None


def _last_nonempty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line:
            return line
    return ""


def _generate_web_bench_html(out_dir: Path, tech: Tech, data_file: Path, data_fmt: str) -> Path:
    script = None
    if tech.tool_id == "plotly":
        script = REPO_ROOT / "scripts" / "web" / "gen_plotly_html_v2.py"
    elif tech.tool_id == "d3":
        script = REPO_ROOT / "scripts" / "web" / "gen_d3_html.py"
    if script is None or not script.exists():
        raise RuntimeError("Web generator script not found.")

    temp_root = out_dir / "_gen"
    cmd = [
        sys.executable,
        str(script),
        "--bench-id",
        "TFFR",
        "--format",
        data_fmt,
        "--file",
        str(data_file),
        "--out-root",
        str(temp_root),
        "--tag",
        "benchkitv2",
        "--window-s",
        "60",
        "--n-ch",
        "8",
    ]
    if tech.tool_id == "d3":
        cmd.append("--no-collect")

    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    html_line = _last_nonempty_line(proc.stdout)
    if not html_line:
        raise RuntimeError("Web generator did not print an HTML path.")
    html_path = Path(html_line)
    if not html_path.exists():
        raise RuntimeError(f"Web generator output not found: {html_path}")

    target = out_dir / f"{tech.tool_id}_bench.html"
    target.write_text(html_path.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def _generate_web_html(out_dir: Path, tech: Tech, data_source: tuple[Path, str] | None) -> Path:
    if data_source is not None:
        data_file, data_fmt = data_source
        return _generate_web_bench_html(out_dir, tech, data_file, data_fmt)
    return _generate_web_smoke_html(out_dir, tech)


def run_all(out_root: Path, tech_names: Iterable[str], timeout_ms: int, data_dir: Path | None = None) -> None:
    selected = _techs_by_name(tech_names)
    out_root = out_root.resolve()
    data_dir = data_dir or (REPO_ROOT / "data")
    data_source = _find_data_file(data_dir)

    for tech in selected:
        bench_root = _ensure_dir(out_root / tech.tool_id)
        ok, reason = _modules_available(tech.modules)

        row = _benchmark_import(tech)
        row["availability"] = reason
        _write_csv_row(bench_root / "import.csv", row)

        if tech.kind == "web":
            try:
                html_path = _generate_web_html(bench_root, tech, data_source)
                collect_html_console(
                    html_path,
                    out_json=bench_root / "bench.json",
                    out_csv=bench_root / "web_console.csv",
                    timeout_ms=timeout_ms,
                )
            except Exception as exc:  # noqa: BLE001 - capture playright missing
                _write_csv_row(
                    bench_root / "web_console.csv",
                    {
                        "timestamp_start": _utc_stamp(),
                        "timestamp_end": _utc_stamp(),
                        "duration_s": "0.000000",
                        "bench_id": "web_smoke",
                        "tool_id": tech.tool_id,
                        "status": "fail",
                        "detail": _safe_value(f"{type(exc).__name__}: {exc}"),
                    },
                )
