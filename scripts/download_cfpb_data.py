"""
Pull a random sample of CFPB consumer complaints with narratives.

Strategy:
  1. Stream-download the bulk CSV ZIP from files.consumerfinance.gov.
  2. Unzip the inner CSV (~5 GB uncompressed).
  3. Stream-parse it, applying filters:
       - narrative present
       - date_received >= configured cutoff
  4. Reservoir-sample down to --limit rows so we get a uniform random
     subset without holding the full table in memory.
  5. Write to the output CSV with our normalized column names so the
     ingestion service can ignore source-format quirks.

Usage:
    python scripts/download_cfpb_data.py
    python scripts/download_cfpb_data.py --limit 50000 --seed 42
    python scripts/download_cfpb_data.py --keep-zip --keep-extract
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import shutil
import sys
import time
import zipfile
from pathlib import Path

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cfpb_download")

# Bulk CSV ZIP — Akamai-fronted, refreshed weekly, ~1.8 GB compressed.
SOURCE_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"

# CFPB source columns → our normalized names.
COLUMN_MAP = {
    "Complaint ID": "complaint_id",
    "Consumer complaint narrative": "consumer_complaint_narrative",
    "Product": "product",
    "Sub-product": "sub_product",
    "Issue": "issue",
    "Sub-issue": "sub_issue",
    "Company": "company",
    "Company response to consumer": "company_response",
    "State": "state",
    "Date received": "date_received",
}
OUTPUT_FIELDS = list(COLUMN_MAP.values())

DEFAULT_OUTPUT = Path("fine_tuning/data/raw/cfpb_200k.csv")
DEFAULT_WORKDIR = Path("fine_tuning/data/raw/_cfpb_workdir")
DEFAULT_LIMIT = 200_000
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_SEED = 17

CHUNK = 1 << 20  # 1 MB streaming chunks
PROGRESS_EVERY = 200_000  # rows


def download(url: str, dest: Path) -> None:
    """Stream the source ZIP to disk with a periodic MB progress line."""
    if dest.exists() and dest.stat().st_size > 0:
        log.info(
            "zip already present at %s (%.1f MB) — skipping download",
            dest,
            dest.stat().st_size / 1e6,
        )
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=None) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        log.info("downloading %s (%.1f MB)", url, total / 1e6)
        downloaded = 0
        last_logged_pct = -1
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes(CHUNK):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = int(downloaded * 100 / total)
                    if pct - last_logged_pct >= 5:
                        log.info("  %d%% (%.1f MB)", pct, downloaded / 1e6)
                        last_logged_pct = pct
    tmp.rename(dest)
    log.info("downloaded %.1f MB to %s", dest.stat().st_size / 1e6, dest)


def extract(zip_path: Path, work_dir: Path) -> Path:
    """Extract the single CSV inside the ZIP, return its path."""
    work_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        candidates = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not candidates:
            raise RuntimeError(f"no CSV found inside {zip_path}")
        if len(candidates) > 1:
            log.warning("multiple CSVs in zip — using %s", candidates[0])
        csv_name = candidates[0]
        out = work_dir / Path(csv_name).name
        if out.exists() and out.stat().st_size > 0:
            log.info("extracted CSV already at %s — skipping", out)
            return out
        log.info("extracting %s ...", csv_name)
        with zf.open(csv_name) as src, out.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=CHUNK)
    log.info("extracted %.1f MB to %s", out.stat().st_size / 1e6, out)
    return out


def reservoir_filter_sample(
    csv_path: Path,
    limit: int,
    start_date: str,
    seed: int,
) -> tuple[list[dict[str, str]], int, int]:
    """Single-pass reservoir sample of rows that pass the filter.

    Returns (sampled_rows, total_scanned, total_passing).
    """
    rng = random.Random(seed)
    reservoir: list[dict[str, str]] = []
    scanned = 0
    passing = 0

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = [k for k in COLUMN_MAP if k not in reader.fieldnames]
        if missing:
            raise RuntimeError(f"source CSV missing expected columns: {missing}")

        for row in reader:
            scanned += 1
            narrative = row.get("Consumer complaint narrative") or ""
            date_recv = row.get("Date received") or ""
            if not narrative.strip():
                continue
            if date_recv < start_date:  # ISO strings sort correctly
                continue
            passing += 1
            normalized = {dst: row.get(src, "") for src, dst in COLUMN_MAP.items()}

            if len(reservoir) < limit:
                reservoir.append(normalized)
            else:
                j = rng.randint(0, passing - 1)
                if j < limit:
                    reservoir[j] = normalized

            if scanned % PROGRESS_EVERY == 0:
                log.info("scanned=%d passing=%d reservoir=%d", scanned, passing, len(reservoir))

    return reservoir, scanned, passing


def write_output(rows: list[dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="target row count")
    p.add_argument("--start-date", default=DEFAULT_START_DATE, help="YYYY-MM-DD lower bound")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="final CSV path")
    p.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR, help="staging dir for zip/csv")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="reservoir sampling seed")
    p.add_argument("--keep-zip", action="store_true", help="don't delete source zip")
    p.add_argument("--keep-extract", action="store_true", help="don't delete extracted CSV")
    args = p.parse_args()

    started = time.monotonic()

    zip_path = args.workdir / "complaints.csv.zip"
    download(SOURCE_URL, zip_path)

    csv_path = extract(zip_path, args.workdir)

    log.info("sampling up to %d rows with seed=%d ...", args.limit, args.seed)
    sampled, scanned, passing = reservoir_filter_sample(
        csv_path,
        args.limit,
        args.start_date,
        args.seed,
    )
    log.info(
        "scan complete: %d total, %d passed filters, %d sampled", scanned, passing, len(sampled)
    )

    write_output(sampled, args.output)
    log.info("wrote %d rows to %s", len(sampled), args.output)

    if not args.keep_zip:
        zip_path.unlink(missing_ok=True)
        log.info("removed source zip")
    if not args.keep_extract:
        csv_path.unlink(missing_ok=True)
        log.info("removed extracted csv")
    if not args.keep_zip and not args.keep_extract:
        try:
            args.workdir.rmdir()
        except OSError:
            pass

    log.info("done in %.1fs", time.monotonic() - started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
