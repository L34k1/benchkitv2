from __future__ import annotations

import sys

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchkit.common import ensure_dir, write_json


def main() -> None:
    p = argparse.ArgumentParser(description="Collect BENCH_JSON from any benchmark HTML via Playwright.")
    p.add_argument("--html", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--console-json", type=Path)
    p.add_argument("--console-csv", type=Path)
    p.add_argument("--headed", action="store_true", help="Launch browser in headed mode.")
    p.add_argument("--timeout-ms", type=int, default=20000)
    args = p.parse_args()

    ensure_dir(args.out.parent)
    if args.console_json:
        ensure_dir(args.console_json.parent)
    if args.console_csv:
        ensure_dir(args.console_csv.parent)

    from playwright.sync_api import sync_playwright

    bench_json: Optional[dict] = None
    errors = []
    console_messages = []
    console_rows = []

    env_headed = os.environ.get("BENCHKIT_COLLECT_HEADED", "").strip().lower() in {"1", "true", "yes"}
    headless = not (args.headed or env_headed)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless, args=["--use-gl=swiftshader", "--enable-webgl"])
        page = browser.new_page()

        def on_console(msg):
            nonlocal bench_json
            ts_ms = int(time.time() * 1000)
            try:
                msg_type = msg.type
            except Exception:
                msg_type = "log"
            try:
                text_attr = getattr(msg, "text", "")
                txt = text_attr() if callable(text_attr) else text_attr
            except Exception:
                txt = str(msg)
            console_rows.append(
                {
                    "ts_ms": ts_ms,
                    "type": msg_type,
                    "text": txt,
                }
            )
            console_messages.append(f"{msg_type}: {txt}")
            if "BENCH_JSON" in txt:
                try:
                    payload = txt.split("BENCH_JSON", 1)[1]
                    if payload.startswith(":"):
                        payload = payload[1:]
                    bench_json = json.loads(payload)
                except Exception as exc:
                    errors.append(f"Failed to parse BENCH_JSON: {exc}")

        page.on("console", on_console)
        page.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))
        page.goto(args.html.resolve().as_uri())

        t0 = time.time()
        while bench_json is None and (time.time() - t0) * 1000 < args.timeout_ms:
            page.wait_for_timeout(50)

        browser.close()

    if args.console_json:
        args.console_json.write_text(json.dumps(console_rows, indent=2), encoding="utf-8")
    if args.console_csv:
        with args.console_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["ts_ms", "type", "text"])
            w.writeheader()
            for row in console_rows:
                w.writerow(row)

    if bench_json is None:
        write_json(
            args.out,
            {
                "error": "BENCH_JSON not captured",
                "errors": errors,
                "console": console_messages[-50:],
                "html": str(args.html),
            },
        )
        raise SystemExit("BENCH_JSON not captured. Open the HTML and copy from DevTools console.")

    write_json(args.out, bench_json)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
