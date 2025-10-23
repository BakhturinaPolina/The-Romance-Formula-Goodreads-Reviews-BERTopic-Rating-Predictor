#!/usr/bin/env python3
"""
Standalone MD5 Downloader (no package dependencies)
---------------------------------------------------
Reads a CSV with MD5 hashes and downloads books using Anna's Archive API.
No heavy dependencies - just requests, pandas, and the API client.
"""

import argparse
import csv
import json
import logging
import os
import pathlib
import sys
from datetime import datetime
from typing import Dict, List, Optional

# Add the book_download directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src" / "book_download"))

from anna_api_client import AnnaAPIClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("standalone_downloader")

MD5_COLUMNS = ("md5_hash", "md5", "hash")


def safe_filename(text: str, max_len: int = 100) -> str:
    """Return *text* converted to a safe filename (ASCII, no spaces)."""
    allowed = "-_.() "
    cleaned = "".join(c for c in text if c.isalnum() or c in allowed).rstrip()
    return cleaned.replace(" ", "_")[:max_len] or "book"


def process_csv(csv_path: pathlib.Path, out_dir: pathlib.Path, client: AnnaAPIClient) -> Dict:
    """Download all MD5s found in *csv_path* into *out_dir*."""
    logger.info("Reading %s", csv_path)
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        logger.warning("CSV appears empty – nothing to do.")
        return {"processed": 0, "downloaded": 0, "skipped": 0}

    # Check MD5 column availability
    if not any(col in reader.fieldnames for col in MD5_COLUMNS):
        raise ValueError(
            f"CSV must contain at least one MD5 column: {', '.join(MD5_COLUMNS)}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    log_records: List[Dict] = []
    downloaded = skipped = 0

    for idx, row in enumerate(rows, start=1):
        md5 = next((row.get(col, "").strip().lower() for col in MD5_COLUMNS if row.get(col)), "")
        title = row.get("title", "unknown")
        author = row.get("author_name", "unknown")

        if not md5 or len(md5) != 32:
            logger.info("[%d/%d] skip (no valid md5): %s — %s", idx, len(rows), title, author)
            skipped += 1
            continue

        # Construct filename so multiple editions don't clash
        filename = f"{safe_filename(title)}_{md5}.epub"
        logger.info("[%d/%d] Downloading %s — %s (%s)…", idx, len(rows), title, author, md5)
        result = client.download_book(md5, filename=filename, download_dir=str(out_dir))

        success = result.get("success", False)
        if success:
            downloaded += 1
            logger.info("  → %s (%d bytes)", result["filepath"], result.get("file_size", 0))
        else:
            logger.warning("  !! failed: %s", result.get("message", "unknown error"))

        log_records.append(
            {
                "idx": idx,
                "title": title,
                "author": author,
                "md5": md5,
                "success": success,
                "filepath": result.get("filepath"),
                "error": result.get("message"),
                "timestamp": datetime.now().isoformat(),
            }
        )

    # Persist a small machine-readable log next to CSV
    log_path = csv_path.with_suffix(".download_log.json")
    with log_path.open("w", encoding="utf-8") as fp:
        json.dump(log_records, fp, indent=2)
    logger.info("Wrote session log → %s", log_path)

    summary = {"processed": len(rows), "downloaded": downloaded, "skipped": skipped}
    logger.info("Summary: %s", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone batch-download books from Anna's Archive using existing MD5 hashes.")
    parser.add_argument("--csv", required=True, type=pathlib.Path, help="Input CSV file with md5 column(s).")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("organized_outputs/epub_downloads"), help="Output directory (default: organized_outputs/epub_downloads)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Ensure secret key is present
    if not os.getenv("ANNAS_SECRET_KEY"):
        logger.error("Environment variable ANNAS_SECRET_KEY not set – export it before running.")
        return 1

    client = AnnaAPIClient()
    try:
        process_csv(args.csv, args.out, client)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Fatal error: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
