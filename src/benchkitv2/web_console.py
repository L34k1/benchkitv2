"""Capture BENCH_JSON from HTML console and write CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


def _ensure_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _as_csv_rows(payload: object) -> Iterable[dict[str, object]]:
    if isinstance(payload, list):
        return [row if isinstance(row, dict) else {"value": row} for row in payload]
    if isinstance(payload, dict):
        return [payload]
    return [{"value": payload}]


def _stringify(value: object) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    _ensure_path(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _stringify(row.get(key, "")) for key in keys})


def _collect_console_payload(html_path: Path, timeout_ms: int) -> object:
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception as exc:  # noqa: BLE001 - used for better error message
        raise RuntimeError("Playwright is required for HTML console capture.") from exc

    captured: list[str] = []

    def on_console(msg) -> None:
        text = msg.text
        if "BENCH_JSON:" in text:
            payload = text.split("BENCH_JSON:", 1)[1].strip()
            captured.append(payload)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.on("console", on_console)
        page.goto(html_path.resolve().as_uri())
        page.wait_for_timeout(timeout_ms)
        browser.close()

    if not captured:
        raise RuntimeError("BENCH_JSON not captured from console.")
    return json.loads(captured[-1])


def collect_html_console(
    html_path: Path,
    out_json: Path | None = None,
    out_csv: Path | None = None,
    timeout_ms: int = 8000,
) -> object:
    payload = _collect_console_payload(html_path, timeout_ms)

    if out_json is None:
        out_json = html_path.with_suffix(".json")
    if out_csv is None:
        out_csv = html_path.with_suffix(".csv")

    _ensure_path(out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(out_csv, _as_csv_rows(payload))
    return payload
