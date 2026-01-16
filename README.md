# benchkitv2

A minimal, runnable Python CLI scaffold for an EEG visualization benchmarking suite.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
benchkitv2
benchkitv2 hello
python -m benchkitv2.cli
```

## Install profiles

```bash
pip install -e ".[core]"
pip install -e ".[desktop]"
pip install -e ".[web]"
pip install -e ".[io]"
pip install -e ".[all]"
```

## Relevant technologies

**Desktop visualization**
- **PyQtGraph**: Qt-based, high-performance 2D plotting for fast waveform rendering.
- **VisPy**: GPU-accelerated visualization built on OpenGL for large datasets.
- **MNE Raw.plot**: Reference scientific EEG viewer; widely used even if slower than GPU stacks.
- **fastplotlib**: Modern GPU viewer; alpha APIs are still evolving.
- **Datoviz**: GPU renderer with potential platform/toolchain constraints; tracked but not installed by default.

**Web visualization**
- **Plotly**: Browser-based interactive plotting with a mature Python API.
- **D3.js**: Custom SVG/canvas rendering; will be embedded via generated HTML that pulls D3 from a CDN.
- **Playwright**: Automation harness for browser performance capture.

**Data formats / IO**
- **EDF** via pyedflib or MNE.
- **NWB** via pynwb + h5py.
- **Neo** as an alternate IO abstraction.

## Development

```bash
python -m unittest -v
```

## Notes / constraints

- GUI stacks require a working display environment (X11/Wayland/Quartz).
- PyQt6 is the selected Qt binding for the desktop stack because it provides widely available wheels and strong compatibility with PyQtGraph and VisPy.
- Some GPU/Qt stacks can fail on Windows due to driver/toolchain mismatches; when that happens, skip the affected profile (e.g., install only `.[core]` or `.[web]`) and capture failures in benchmark metadata.
- Datoviz is not listed in the install profiles because it is not reliably pip-installable across platforms; track it separately if you need GPU benchmarking coverage.
