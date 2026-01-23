# benchkitv2

Benchkitv2 is an EEG visualization benchmarking suite. It runs IO and rendering benchmarks
across multiple desktop and web stacks, writes per-run artifacts, and produces analysis-ready
CSV outputs.

This repo is organized around a single orchestrator (`run_all.py`) plus individual bench scripts
under `scripts/`. Windows entrypoints wrap setup and execution in `.bat` files.

## Repository layout

- `run_all.bat` / `run_all.py`: end-to-end orchestrator.
- `scripts/desktop`, `scripts/web`, `scripts/io`: per-tool benchmarks.
- `scripts/analysis`: output normalization + aggregation.
- `scripts/data`: synthetic data generation helpers.
- `benchkit/`: shared helpers and output contract.
- `data/`: input EDF/NWB files (local datasets, ignored by git).
- `outputs/`: benchmark outputs and aggregated CSVs (ignored by git).
- `docs/`: architecture notes, lexicon, task lists.

## Quickstart (Windows)

Run everything once (creates `.venv`, installs deps, runs benches, aggregates outputs):

```bat
run_all.bat
```

Pass any `run_all.py` flags through the batch file:

```bat
run_all.bat --data-dir data --out-root outputs --modes IO TFFR A1 A2 --runs 1
```

## Quickstart (manual, any OS)

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
# or: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r scripts/desktop/requirements-desktop.txt
python -m pip install -r scripts/web/requirements-web.txt
python -m playwright install chromium
python run_all.py --runs 1
```

## Dependencies and profiles

The orchestrator uses the requirements files under `scripts/`:

- `scripts/desktop/requirements-desktop.txt` (PyQt5, PyQtGraph, VisPy, MNE, Datoviz, fastplotlib, etc.)
- `scripts/web/requirements-web.txt` (Playwright)

The `pyproject.toml` extras are for the minimal `benchkitv2` CLI scaffold and are not required
to run `run_all.py`. If you use the CLI extras, note that the pyqtgraph benches use PyQt5
internally, while the extras currently specify PyQt6.

Datoviz is included in the desktop requirements. If it is not installable on your platform,
remove it from `scripts/desktop/requirements-desktop.txt` and skip it with
`BENCHKIT_SKIP_TOOLS=VIS_DATOVIZ`.

## Data inputs

Place EDF/NWB files anywhere under `data/`. The orchestrator scans recursively.

Generate a small synthetic file:

```bat
gen_synth_one.bat EDF
```

This writes to `data/_synth/` by default.

## Running benchmarks

### Orchestrator

`run_all.py` executes all tool/bench combinations and writes outputs to
`outputs/<BENCH_ID>/<TOOL_ID>/<tag>/`.

Key flags:

- `--data-dir`: root folder to scan for EDF/NWB data.
- `--out-root`: output folder (default `outputs`).
- `--modes`: subset of `IO`, `TFFR`, `A1`, `A2`.
- `--tools`: subset of tool IDs (see `docs/LEXICON.md`).
- `--runs`, `--windows`, `--channels`, `--sequences`, `--overlays`, `--cadence-ms`

Example (desktop only, 1 run):

```bash
python run_all.py --data-dir data --out-root outputs --modes TFFR A1 A2 --tools VIS_PYQTGRAPH VIS_VISPY
```

Plotly is intentionally excluded from the default `--tools all` set. To run Plotly, pass it
explicitly:

```bash
python run_all.py --tools VIS_PLOTLY
```

Plotly benches require `plotly` installed (it is not in `requirements-web.txt`), so install it
separately or via `pip install -e ".[web]"`.

### Smoke scripts

- `smoke_all_techs_8ch_600s.bat`: one-pass orchestrator run on 8ch/600s.
- `smoke_all_outputs.bat`: runs many scripts and validates outputs with `benchkit.smoke`.
- `run_one_each.bat`: runs a single job per tool; update `DATA_FILE` to a real EDF/NWB file.
- `run_smoke_check.bat`: ensures every script responds to `--help`.

## Outputs and analysis

Per-run outputs live under `outputs/<BENCH_ID>/<TOOL_ID>/<tag>/`:

- `manifest.json`
- `steps.csv` (A1/A2 benches)
- `tffr.csv` (TFFR bench)
- `summary.json` and optional `*summary*.csv`

Aggregate outputs:

```bash
python scripts/analysis/unify_outputs_csv.py --out-root outputs --out-dir outputs/_aggregate/unified
python scripts/analysis/aggregate_by_tool_phase.py --unified-dir outputs/_aggregate/unified --out-dir outputs/_aggregate/by_tool_phase
python scripts/analysis/aggregate_by_tool_window_channels.py --unified-dir outputs/_aggregate/unified --out-dir outputs/_aggregate/by_tool_window_channels
```

Optional merged raw outputs:

```bash
python scripts/merge_outputs_csv.py --out-root outputs --out-dir outputs/_orchestration
```

## CLI scaffold (benchkitv2)

`benchkitv2` is a minimal CLI for quick import checks and web console capture:

```bash
pip install -e .
benchkitv2 deps
benchkitv2 run --out-root outputs
benchkitv2 collect-html --html outputs/plotly/plotly_bench.html
```

## Notes

- GUI stacks require an active display environment.
- Some GPU stacks can fail on Windows due to driver/toolchain mismatches; skip tools as needed
  with `BENCHKIT_SKIP_TOOLS`.
- See `docs/ARCHITECTURE.md`, `docs/LEXICON.md`, and `docs/TASKLIST_BENCH_OUTPUTS.md` for deeper
  references.
