"""Command line interface for benchkitv2."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from benchkitv2.runner import run_all
from benchkitv2.web_console import collect_html_console

DEPENDENCY_PROFILES = {
    "core": [
        "numpy==2.2.6",
        "pandas==2.2.3",
    ],
    "desktop": [
        "pyqtgraph==0.13.7",
        "PyQt6==6.7.1",
        "vispy==0.14.3",
        "mne==1.7.1",
        "matplotlib==3.9.2",
        "fastplotlib==0.6.0",
    ],
    "web": [
        "plotly==5.24.1",
        "playwright==1.47.0",
    ],
    "io": [
        "pyedflib==0.1.38",
        "pynwb==2.8.2",
        "neo==0.13.2",
        "h5py==3.12.1",
    ],
    "all": [
        "core + desktop + web + io",
    ],
}


def _format_profiles(profiles: dict[str, Iterable[str]]) -> str:
    lines = ["Dependency profiles:"]
    for name, deps in profiles.items():
        lines.append(f"- {name}:")
        for dep in deps:
            lines.append(f"  - {dep}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchkitv2")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("hello", help="Print a friendly greeting.")
    subparsers.add_parser("deps", help="List dependency profiles.")
    run_parser = subparsers.add_parser("run", help="Run quick benchmarks for all techs.")
    run_parser.add_argument("--out-root", type=Path, default=Path("outputs"))
    run_parser.add_argument("--techs", nargs="+", default=["all"])
    run_parser.add_argument("--timeout-ms", type=int, default=8000)
    run_parser.add_argument("--data-dir", type=Path, default=Path("data"))

    collect_parser = subparsers.add_parser(
        "collect-html", help="Collect BENCH_JSON from HTML and write CSV."
    )
    collect_parser.add_argument("--html", type=Path, required=True)
    collect_parser.add_argument("--out-json", type=Path, default=None)
    collect_parser.add_argument("--out-csv", type=Path, default=None)
    collect_parser.add_argument("--timeout-ms", type=int, default=8000)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        print("Hello, world!")
        return 0

    if args.command == "hello":
        print("Hello, world!")
        return 0

    if args.command == "deps":
        print(_format_profiles(DEPENDENCY_PROFILES))
        return 0
    if args.command == "run":
        run_all(args.out_root, args.techs, args.timeout_ms, data_dir=args.data_dir)
        return 0
    if args.command == "collect-html":
        collect_html_console(
            args.html,
            out_json=args.out_json,
            out_csv=args.out_csv,
            timeout_ms=args.timeout_ms,
        )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
