#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def slugify(s: str) -> str:
    s = s.replace("\\", "/")
    s = re.sub(r"[^A-Za-z0-9._/-]+", "_", s)
    s = s.replace("/", "__")
    return s.strip("_")


def read_matrix(csv_path: Path) -> pd.DataFrame:
    m = pd.read_csv(csv_path, index_col=0)
    m.index = pd.to_numeric(m.index, errors="coerce")
    m.columns = pd.to_numeric(m.columns, errors="coerce")
    m = m.sort_index(axis=0).sort_index(axis=1)
    m = m.loc[~m.index.isna(), ~pd.isna(m.columns)]
    return m


def metric_unit(metric_name: str) -> str:
    if "latency" in metric_name:
        return "ms"
    if "updates_per_s" in metric_name:
        return "updates/s"
    return ""


def safe_figsize(n_rows: int, n_cols: int) -> tuple[float, float]:
    w = max(5.0, min(14.0, 0.55 * n_cols + 3.0))
    h = max(4.0, min(12.0, 0.45 * n_rows + 3.0))
    return (w, h)


def save_all_formats(fig, out_base: Path, formats: list[str], dpi: int = 220) -> list[Path]:
    out_paths = []
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        p = out_base.with_suffix(f".{fmt}")
        if fmt.lower() in ("png", "jpg", "jpeg", "webp"):
            fig.savefig(p, dpi=dpi, bbox_inches="tight")
        else:
            fig.savefig(p, bbox_inches="tight")
        out_paths.append(p)
    return out_paths


def plot_heatmap(
    m: pd.DataFrame,
    title: str,
    metric_name: str,
    out_base: Path,
    formats: list[str],
) -> list[Path]:
    data = m.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(data)

    fig, ax = plt.subplots(figsize=safe_figsize(m.shape[0], m.shape[1]))
    im = ax.imshow(masked, aspect="auto", interpolation="nearest")

    ax.set_title(title)
    ax.set_xlabel("window_s")
    ax.set_ylabel("n_channels")

    cols = m.columns.to_list()
    rows = m.index.to_list()

    def tick_positions(n: int, max_ticks: int = 12) -> list[int]:
        if n <= max_ticks:
            return list(range(n))
        step = max(1, n // max_ticks)
        pos = list(range(0, n, step))
        if pos[-1] != n - 1:
            pos.append(n - 1)
        return pos

    xt = tick_positions(len(cols))
    yt = tick_positions(len(rows))

    ax.set_xticks(xt)
    ax.set_xticklabels([str(cols[i]) for i in xt], rotation=45, ha="right")
    ax.set_yticks(yt)
    ax.set_yticklabels([str(rows[i]) for i in yt])

    unit = metric_unit(metric_name)
    cbar = fig.colorbar(im, ax=ax)
    if unit:
        cbar.set_label(unit)

    paths = save_all_formats(fig, out_base, formats)
    plt.close(fig)
    return paths


def pick_three(values: list[float]) -> list[float]:
    if not values:
        return []
    vals = sorted(values)
    if len(vals) <= 2:
        return vals
    mid = vals[len(vals) // 2]
    return sorted(set([vals[0], mid, vals[-1]]))


def plot_slices(
    m: pd.DataFrame,
    title: str,
    metric_name: str,
    out_base_prefix: Path,
    formats: list[str],
) -> list[Path]:
    out_paths: list[Path] = []
    unit = metric_unit(metric_name)

    if m.shape[0] >= 1 and m.shape[1] >= 2:
        sel_ch = pick_three([float(x) for x in m.index.to_list()])
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        for ch in sel_ch:
            if ch in m.index:
                ax.plot(
                    m.columns.to_numpy(dtype=float),
                    m.loc[ch].to_numpy(dtype=float),
                    marker="o",
                    label=f"ch={ch}",
                )
        ax.set_title(f"{title} — vs window")
        ax.set_xlabel("window_s")
        ax.set_ylabel(f"{metric_name}" + (f" ({unit})" if unit else ""))
        ax.legend()
        out_paths += save_all_formats(
            fig,
            out_base_prefix.with_name(out_base_prefix.name.replace("PREFIX", "slice_window")),
            formats,
        )
        plt.close(fig)

    if m.shape[0] >= 2 and m.shape[1] >= 1:
        sel_w = pick_three([float(x) for x in m.columns.to_list()])
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        for w in sel_w:
            if w in m.columns:
                ax.plot(
                    m.index.to_numpy(dtype=float),
                    m[w].to_numpy(dtype=float),
                    marker="o",
                    label=f"w={w}",
                )
        ax.set_title(f"{title} — vs channels")
        ax.set_xlabel("n_channels")
        ax.set_ylabel(f"{metric_name}" + (f" ({unit})" if unit else ""))
        ax.legend()
        out_paths += save_all_formats(
            fig,
            out_base_prefix.with_name(out_base_prefix.name.replace("PREFIX", "slice_channels")),
            formats,
        )
        plt.close(fig)

    return out_paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="out/matrices", help="Root folder containing matrix CSVs")
    ap.add_argument("--out", type=str, default="out/figures", help="Output folder for generated figures")
    ap.add_argument("--formats", nargs="+", default=["svg", "png"], help="File formats to write")
    ap.add_argument("--include", nargs="*", default=[], help="Optional substrings to include")
    ap.add_argument("--exclude", nargs="*", default=["index", "summary"], help="Optional substrings to exclude")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    csvs = sorted(root.glob("**/*.csv"))

    produced: list[Path] = []
    skipped = 0

    for csv_path in csvs:
        rel = csv_path.relative_to(root).as_posix()

        if any(x in rel for x in args.exclude):
            skipped += 1
            continue
        if args.include and not any(x in rel for x in args.include):
            skipped += 1
            continue

        try:
            m = read_matrix(csv_path)
            if m.empty:
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue

        metric_name = csv_path.stem
        rel_no_ext = rel.replace(".csv", "")
        title = rel_no_ext.replace("/", " / ")

        parts = csv_path.relative_to(root).parts
        if len(parts) >= 3:
            bench_id, tool_id = parts[0], parts[1]
            base_dir = out / slugify(bench_id) / slugify(tool_id)
            base = base_dir / metric_name
        else:
            base = out / slugify(rel_no_ext)

        produced += plot_heatmap(
            m=m,
            title=title,
            metric_name=metric_name,
            out_base=base.with_name("heatmap__" + base.name),
            formats=args.formats,
        )

        produced += plot_slices(
            m=m,
            title=title,
            metric_name=metric_name,
            out_base_prefix=base.with_name("PREFIX__" + base.name),
            formats=args.formats,
        )

    out.mkdir(parents=True, exist_ok=True)
    index_csv = out / "_generated_index.csv"
    pd.DataFrame({"figure_path": [str(p) for p in produced]}).to_csv(index_csv, index=False)

    print(f"[OK] Figures written: {len(produced)}")
    print(f"[OK] Index: {index_csv}")
    print(f"[INFO] CSVs scanned: {len(csvs)} | skipped: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
