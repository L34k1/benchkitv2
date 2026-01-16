"""Command line interface for benchkitv2."""

from __future__ import annotations

import argparse
from typing import Iterable

DEPENDENCY_PROFILES = {
    "core": [
        "numpy==2.1.4",
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

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
