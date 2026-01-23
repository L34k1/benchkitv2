from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_dict(prefix: str, payload: Dict[str, object]) -> Dict[str, object]:
    flat: Dict[str, object] = {}
    for key, value in payload.items():
        out_key = f"{prefix}{key}"
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat[f"{out_key}_{sub_key}"] = sub_value
        else:
            flat[out_key] = value
    return flat


def ensure_str(value: object) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=True)
    return "" if value is None else str(value)


def find_runs(out_root: Path) -> Iterable[Tuple[Path, Dict[str, object]]]:
    for manifest_path in out_root.rglob("manifest.json"):
        if "_orchestration" in manifest_path.parts:
            continue
        run_dir = manifest_path.parent
        yield run_dir, load_json(manifest_path)


def collect_csv_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ensure_str(row.get(k)) for k in keys})


def main() -> None:
    p = argparse.ArgumentParser(description="Merge benchmark CSVs with metadata from outputs/*/manifest.json.")
    p.add_argument("--out-root", type=Path, default=Path("outputs"))
    p.add_argument("--out-dir", type=Path)
    args = p.parse_args()

    out_root = args.out_root
    out_dir = args.out_dir or (out_root / "_orchestration")
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_steps: List[Dict[str, object]] = []
    merged_tffr: List[Dict[str, object]] = []
    merged_summary_csv: List[Dict[str, object]] = []
    merged_summary_json: List[Dict[str, object]] = []

    for run_dir, manifest in find_runs(out_root):
        rel_dir = str(run_dir.relative_to(out_root))
        base = dict(manifest)
        base["out_dir"] = rel_dir
        base["tag"] = run_dir.name

        meta = {}
        if isinstance(manifest.get("extra"), dict):
            extra = manifest.get("extra") or {}
            if isinstance(extra, dict) and isinstance(extra.get("env"), dict):
                meta.update(flatten_dict("env_", extra.get("env") or {}))

        # steps.csv
        steps_path = run_dir / "steps.csv"
        if steps_path.exists():
            for row in collect_csv_rows(steps_path):
                merged_steps.append({**base, **meta, **row})

        # tffr.csv
        tffr_path = run_dir / "tffr.csv"
        if tffr_path.exists():
            for row in collect_csv_rows(tffr_path):
                merged_tffr.append({**base, **meta, **row})

        # summary.csv (or *summary*.csv)
        for summary_csv in run_dir.glob("*summary*.csv"):
            for row in collect_csv_rows(summary_csv):
                merged_summary_csv.append(
                    {**base, **meta, "summary_csv": summary_csv.name, **row}
                )

        # summary.json
        summary_json_path = run_dir / "summary.json"
        if summary_json_path.exists():
            summary = load_json(summary_json_path)
            meta_block = {}
            if isinstance(summary.get("meta"), dict):
                meta_block = flatten_dict("meta_", summary.get("meta") or {})
            summary_row = {**base, **meta, **summary, **meta_block}
            summary_row.pop("meta", None)
            merged_summary_json.append(summary_row)

    write_csv(out_dir / "steps_merged.csv", merged_steps)
    write_csv(out_dir / "tffr_merged.csv", merged_tffr)
    write_csv(out_dir / "summary_csv_merged.csv", merged_summary_csv)
    write_csv(out_dir / "summary_json_merged.csv", merged_summary_json)


if __name__ == "__main__":
    main()
