#!/usr/bin/env python3
"""
Search and Download Script
-------------------------
Reads a CSV with title/author columns, searches Anna's Archive for each book,
and downloads the best match. Uses the search_md5() method for title-based retrieval.
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
logger = logging.getLogger("search_and_download")

def safe_filename(text: str, max_len: int = 100) -> str:
    """Return *text* converted to a safe filename (ASCII, no spaces)."""
    allowed = "-_.() "
    cleaned = "".join(c for c in text if c.isalnum() or c in allowed).rstrip()
    return cleaned.replace(" ", "_")[:max_len] or "book"


def process_csv(csv_path: pathlib.Path, out_dir: pathlib.Path, client: AnnaAPIClient) -> Dict:
    """Search and download all books found in *csv_path* into *out_dir*."""
    logger.info("Reading %s", csv_path)
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        logger.warning("CSV appears empty – nothing to do.")
        return {"processed": 0, "downloaded": 0, "skipped": 0, "not_found": 0}

    # Check required columns
    required_cols = ['title', 'author_name']
    missing_cols = [col for col in required_cols if col not in reader.fieldnames]
    if missing_cols:
        raise ValueError(f"CSV must contain columns: {missing_cols}")

    out_dir.mkdir(parents=True, exist_ok=True)

    log_records: List[Dict] = []
    downloaded = skipped = not_found = 0

    for idx, row in enumerate(rows, start=1):
        title = row.get('title', '').strip()
        author = row.get('author_name', '').strip()
        work_id = row.get('work_id', idx)
        year = row.get('publication_year', '')

        if not title or not author:
            logger.info("[%d/%d] skip (missing title/author): %s — %s", idx, len(rows), title, author)
            skipped += 1
            continue

        logger.info("[%d/%d] Searching for: '%s' by %s", idx, len(rows), title, author)
        
        # Search for MD5 using title and author
        md5 = client.search_md5(title, author, prefer_exts=("epub", "mobi", "pdf"))
        
        if not md5:
            logger.warning("  !! No match found for: %s by %s", title, author)
            not_found += 1
            log_records.append({
                "idx": idx,
                "work_id": work_id,
                "title": title,
                "author": author,
                "year": year,
                "md5": None,
                "success": False,
                "error": "No match found in Anna's Archive",
                "timestamp": datetime.now().isoformat(),
            })
            continue

        # Download the book
        filename = f"{safe_filename(title)}_{work_id}_{md5}.epub"
        logger.info("  → Found MD5: %s, downloading...", md5)
        
        result = client.download_book(md5, filename=filename, download_dir=str(out_dir))

        success = result.get("success", False)
        if success:
            downloaded += 1
            logger.info("  ✅ Downloaded: %s (%d bytes)", result["filepath"], result.get("file_size", 0))
        else:
            logger.warning("  ❌ Download failed: %s", result.get("message", "unknown error"))

        log_records.append({
            "idx": idx,
            "work_id": work_id,
            "title": title,
            "author": author,
            "year": year,
            "md5": md5,
            "success": success,
            "filepath": result.get("filepath"),
            "error": result.get("message"),
            "timestamp": datetime.now().isoformat(),
        })

    # Persist session log
    log_path = csv_path.with_suffix(".search_download_log.json")
    with log_path.open("w", encoding="utf-8") as fp:
        json.dump(log_records, fp, indent=2)
    logger.info("Wrote session log → %s", log_path)

    summary = {
        "processed": len(rows), 
        "downloaded": downloaded, 
        "skipped": skipped, 
        "not_found": not_found
    }
    logger.info("Summary: %s", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Search and download books from Anna's Archive using title/author.")
    parser.add_argument("--csv", required=True, type=pathlib.Path, help="Input CSV file with title, author_name columns.")
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
