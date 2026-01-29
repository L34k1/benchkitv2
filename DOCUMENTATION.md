# benchkitv2 Documentation

This document describes the benchkitv2 EEG visualization benchmarking suite: its goals, layout, core concepts, orchestration flow, outputs, analysis pipeline, and extension points. It is written from the current repository state and is meant to be the primary on-disk reference for day-to-day usage.

## 1) Overview

benchkitv2 is a reproducible benchmarking pack for EEG visualization stacks. It measures:

- IO: dataset read and decode cost (EDF/NWB backends)
- TFFR: time to first render
- A1_THROUGHPUT: back-to-back interaction throughput
- A2_CADENCED: cadenced interactions and lateness (jank proxy)

It supports desktop (PyQtGraph, VisPy, Datoviz, fastplotlib, MNE RawPlot), web (Plotly, D3 canvas), and IO stacks (pyEDFlib, MNE, Neo, PyNWB, h5py). Results are written in a consistent output contract for downstream analysis.

## 2) Canonical terminology (lexicon)

The canonical IDs live in `benchkit/lexicon.py` and are used throughout outputs, logs, and analysis.

### Bench IDs
- IO
- TFFR
- A1_THROUGHPUT (A1)
- A2_CADENCED (A2)

### Formats
- EDF
- NWB

### Tools (canonical IDs)
- IO_PYEDFLIB, IO_MNE, IO_NEO, IO_PYNWB, IO_H5PY
- VIS_PYQTGRAPH
- VIS_VISPY
- VIS_DATOVIZ
- VIS_FASTPLOTLIB
- VIS_MNE_RAWPLOT
- VIS_PLOTLY
- VIS_D3_CANVAS

### Interaction sequences
- PAN
- ZOOM_IN
- ZOOM_OUT
- PAN_ZOOM

### Overlay and cache state
- OVL_OFF / OVL_ON
- CACHE_WARM / CACHE_COLD

These IDs are required for consistent output naming and for downstream aggregation.

## 3) Repository layout

Top-level:
- `run_all.py`: end-to-end orchestrator (builds job matrix, runs all benches, collects web results).
- `run_all.bat`, `run_one_each.bat`, `run_smoke_check.bat`, `smoke_all_outputs.bat`, `smoke_all_techs_8ch_600s.bat`: Windows wrappers.
- `gen_synth_one.bat`: one-shot synthetic data generation wrapper.
- `README.md`: quickstart and high-level usage.

Core packages:
- `benchkit/`: shared helpers, lexicon, output contract, loaders, stats, and smoke validation.
- `src/benchkitv2/`: minimal CLI scaffold (benchkitv2) for quick checks.

Scripts (the actual benchmarks):
- `scripts/io/`: IO benchmarks.
- `scripts/desktop/`: desktop visualization benchmarks per stack.
- `scripts/web/`: web visualization benchmarks and collectors.
- `scripts/analysis/`: aggregation and matrix/figure generation.
- `scripts/data/`: synthetic data generation.
- `scripts/utils/`: smoke check helpers.

Data and outputs:
- `data/`: local EDF/NWB datasets (not versioned).
- `outputs/`: benchmark outputs (not versioned).
- `outputs/_orchestration/`: orchestrator logs, job logs, and timings.

Docs:
- `docs/ARCHITECTURE.md`, `docs/LEXICON.md`, `docs/STATUS.md`, `docs/TASKLIST_BENCH_OUTPUTS.md`.

Config:
- `configs/coverage_matrix_edf.json`: planning matrix for EDF coverage (not consumed automatically).

## 4) Setup and dependencies

### Python
- Requires Python >= 3.11 (see `pyproject.toml`).

### Recommended venv setup (Windows)
```bat
python -m venv .venv
call .\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r scripts\desktop\requirements-desktop.txt
python -m pip install -r scripts\web\requirements-web.txt
python -m playwright install chromium
```

### Dependency profiles (CLI scaffold)
`benchkitv2` exposes a `deps` subcommand that prints dependency profiles. These profiles map to the extras in `pyproject.toml` but are not required to run `run_all.py`.

### Optional environment variable
- `BENCHKIT_SKIP_TOOLS`: comma/semicolon-separated list of tool IDs to skip (e.g., `VIS_DATOVIZ,VIS_VISPY`).

## 5) Data inputs

Place EDF/NWB files anywhere under `data/` (recursive scan). The orchestrator and IO scripts will pick them up.

Synthetic data helper:
- `scripts/data/gen_synth_one.py` (wrapped by `gen_synth_one.bat`).
- Output defaults to `data/_synth/`.

## 6) Orchestrator: `run_all.py`

### What it does
- Scans `data/` for EDF/NWB files.
- Builds a job matrix across tools, benches, windows, channels, sequences, overlays, cadence, and runs.
- Executes benchmark scripts and captures stdout/stderr to per-job log files.
- For web tools (Plotly, D3), runs a Playwright console collector after the HTML generator.
- Enforces timeouts and heartbeats to prevent hangs.

### Defaults and job matrix
Defaults in `run_all.py`:
- windows: 60, 600, 1800 seconds
- channels: 8, 16, 32, 64
- sequences: PAN, ZOOM_IN, ZOOM_OUT, PAN_ZOOM
- overlays: OVL_OFF, OVL_ON
- cadence_ms: 16.0 (for A2)
- runs: 1

`--load-duration-multiplier` defaults to 10.0 (load window = window * 10).

### Tools
`--tools all` runs all tools except Plotly by default. Plotly must be explicitly requested.

### Job tagging
A unique tag is generated per job based on:
- file name
- format
- window
- load duration
- channels
- sequence
- overlay
- run id
- cadence (A2 only)

This tag is used to create the output folder:
```
outputs/<BENCH_ID>/<TOOL_ID>/<tag>/
```

### Timeouts
Default timeouts in `run_all.py`:
- IO: 120s
- TFFR: 60s
- A1: 180s
- A2: 180s

`--heartbeat-timeout-s` defaults to 15s. Jobs that stop emitting output are terminated and marked failed.

### Orchestrator logs
- `outputs/_orchestration/orchestrator.log`: aggregate log
- `outputs/_orchestration/jobs/<job_id>.log`: per-job log
- `outputs/_orchestration/timings.csv`: timing/exit status for each job

### Example
```bash
python run_all.py --data-dir data --out-root outputs --modes IO TFFR A1 A2 --tools VIS_PYQTGRAPH VIS_VISPY
```

## 7) Output contract

### Directory structure
All runs live under:
```
outputs/<BENCH_ID>/<TOOL_ID>/<tag>/
```

### Required files
- `manifest.json`: required for all benches
- `summary.json` (or `summary.csv` or `*summary*.csv`)
- `steps.csv` for A1/A2
- `tffr.csv` for TFFR

### Canonical step columns
`steps.csv` should include:
- `step_id`
- `latency_ms`
- `noop`
- `status`

Additional columns are allowed (e.g., `lateness_ms`, `command_t_ms`, `visible_t_ms`).

### Manifest contract
`benchkit/output_contract.py` provides `write_manifest_contract(...)` with fields:
- bench_id, tool_id, format, file, window_s, n_channels
- sequence, overlay, run_id, steps_target
- timestamp_utc, extra

### Summary helpers
`benchkit/output_contract.py` provides helpers to write:
- `summary.json` for steps (A1/A2)
- `tffr.csv` + `summary.json` for TFFR

### Smoke checks
`benchkit/smoke.py` validates `steps.csv` and `tffr.csv` shapes and numeric fields.

## 8) Bench scripts

### IO
- `scripts/io/bench_io_v2.py`: EDF/NWB IO benchmark. Writes `summary.json` and `per_file.json` for each tool. Supports:
  - EDF: IO_PYEDFLIB, IO_MNE, IO_NEO
  - NWB: IO_PYNWB, IO_H5PY
- `scripts/io/bench_io.py`: legacy EDF-only benchmark with lazy/preload profiles.

### Desktop visualization
Each script emits the standard output contract:
- `scripts/desktop/bench_pyqtgraph_tffr_v2.py`
- `scripts/desktop/bench_pyqtgraph_A1_throughput_v2.py`
- `scripts/desktop/bench_pyqtgraph_A2_cadenced_v2.py`
- `scripts/desktop/bench_vispy_A1_throughput.py`
- `scripts/desktop/bench_vispy_A2_cadenced.py`
- `scripts/desktop/bench_vispy.py` (multi-bench entrypoint)
- `scripts/desktop/bench_datoviz.py`
- `scripts/desktop/bench_fastplotlib.py`
- `scripts/desktop/bench_fastplotlib_a1.py`, `bench_fastplotlib_a2.py`
- `scripts/desktop/bench_mne_rawplot.py`

The fastplotlib helpers live in `scripts/desktop/_fastplotlib_common.py`.

### Web visualization
- `scripts/web/gen_plotly_html_v2.py`: generate Plotly benchmark HTML, optionally collect via Playwright.
- `scripts/web/gen_d3_html.py`: generate D3 canvas benchmark HTML, optionally collect via Playwright.
- `scripts/web/bench_plotly_*.py`: convenience entrypoints for Plotly A1/A2/TFFR.
- Collectors:
  - `scripts/web/collect_console_playwright.py`
  - `scripts/web/collect_plotly_console_playwright_v2.py`
  - `scripts/web/collect_plotly_console_selenium_v2.py`

Web benches emit HTML, then a Playwright collector extracts `BENCH_JSON` from console logs and writes `bench.json`, `console.json`, `console.csv`, plus `steps.csv` and `summary.json`.

## 9) Analysis pipeline

The analysis scripts consume outputs and produce unified and aggregated CSVs:

1) Unify outputs:
```bash
python scripts/analysis/unify_outputs_csv.py --out-root outputs --out-dir outputs/_aggregate/unified
```

2) Aggregate by tool and phase:
```bash
python scripts/analysis/aggregate_by_tool_phase.py --unified-dir outputs/_aggregate/unified --out-dir outputs/_aggregate/by_tool_phase
```

3) Aggregate by window and channels:
```bash
python scripts/analysis/aggregate_by_tool_window_channels.py --unified-dir outputs/_aggregate/unified --out-dir outputs/_aggregate/by_tool_window_channels
```

4) Build matrices (median metrics per tool/bench/window/channel):
```bash
python scripts/analysis/make_window_channel_matrices.py --unified-dir outputs/_aggregate/unified --out-dir out/matrices
```

5) Plot matrices:
```bash
python scripts/analysis/plot_window_channel_matrices.py --root out/matrices --out out/figures
```

## 10) CLI scaffold: `benchkitv2`

The `benchkitv2` CLI is a minimal scaffold for sanity checks and quick console capture:

- `benchkitv2 hello`: prints Hello, world.
- `benchkitv2 deps`: prints dependency profiles.
- `benchkitv2 run`: runs quick import checks and web console capture.
- `benchkitv2 collect-html`: collects `BENCH_JSON` from a provided HTML file.

The CLI lives in `src/benchkitv2/` and is intentionally small. The real orchestration is in `run_all.py`.

## 11) Testing

Tests use `unittest`:
- `tests/test_cli.py`: CLI module invocation test.
- `tests/test_datoviz_utils.py`: datoviz range utilities.
- `tests/test_fastplotlib_sequences.py`: deterministic sequence generation.
- `tests/e2e/example.spec.js`: Playwright example.

Run unit tests:
```bash
python -m unittest -v
```

## 12) Configuration and planning helpers

- `configs/coverage_matrix_edf.json`: planning matrix for EDF coverage; not consumed by scripts automatically.
- `docs/TASKLIST_BENCH_OUTPUTS.md`: checklist for output contract completion and smoke commands.
- `docs/STATUS.md`: best-effort status snapshot (manual).

## 13) Extending the suite

To add a new visualization tool or benchmark:

1) Add tool ID in `benchkit/lexicon.py`.
2) Implement a new script in `scripts/desktop` or `scripts/web` that writes:
   - `manifest.json`
   - `summary.json`
   - `steps.csv` (A1/A2) or `tffr.csv` (TFFR)
3) Register it in `run_all.py -> build_tools()`.
4) Ensure any web tool emits `BENCH_JSON` for Playwright collection.
5) Update `docs/LEXICON.md` and `docs/TASKLIST_BENCH_OUTPUTS.md` as needed.

## 14) Troubleshooting and caveats

- GUI stacks require an active display environment.
- GPU/OpenGL backends can be fragile on Windows depending on drivers and runtime.
- Plotly is excluded from the default `--tools all` set; pass it explicitly.
- Playwright must be installed and browsers installed for web collectors.
- Some tools allow skipping via `BENCHKIT_SKIP_TOOLS`.

## 15) Quick reference commands

Run everything:
```bat
run_all.bat
```

Run a subset:
```bash
python run_all.py --data-dir data --out-root outputs --modes TFFR A1 A2 --tools VIS_PYQTGRAPH VIS_VISPY
```

Generate a synthetic dataset:
```bat
gen_synth_one.bat EDF
```

Unify outputs:
```bash
python scripts/analysis/unify_outputs_csv.py --out-root outputs --out-dir outputs/_aggregate/unified
```
