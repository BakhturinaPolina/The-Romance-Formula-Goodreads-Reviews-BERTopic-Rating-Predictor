#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep diagnostics for Step-1/2 outputs:
- books_with_shelf_norm.parquet
- shelves_raw_long.parquet
- shelves_canon_long.parquet
- segments_long.parquet
- shelf_canonical.csv
- shelf_segments.csv
- noncontent_shelves.csv
- shelf_alias_candidates.csv (optional)

Emits:
- bridge_qa/diagnostics_summary.json
- bridge_qa/diagnostics_report.txt (human-readable)
- bridge_qa/suspect_examples_*.csv (small samples)
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import unicodedata as ud

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("bridge_diagnostics")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def utcnow_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def ensure_outdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception as e:
        logger.warning("pyarrow read failed (%s), retrying with fastparquet…", e)
        return pd.read_parquet(path, engine="fastparquet")

def to_csv_sample(df: pd.DataFrame, path: Path, n: int = 200) -> None:
    df.head(n).to_csv(path, index=False)

def is_iterableish(x: Any) -> bool:
    return isinstance(x, (list, tuple, np.ndarray, pd.Series, pd.Index))

def safe_len(x: Any) -> int:
    try:
        return len(x) if is_iterableish(x) else 0
    except Exception:
        return 0

def norm_key(s: str) -> str:
    if s is None:
        return ""
    return " ".join(str(s).strip().casefold().split())

def unicode_profile(s: str) -> Dict[str, int]:
    """Rough unicode diagnostics per token."""
    cats = {}
    for ch in s:
        c = ud.category(ch)  # e.g., 'Ll', 'Lu', 'Nd', 'Po', etc.
        cats[c] = cats.get(c, 0) + 1
    return cats

# -----------------------------------------------------------------------------
# Checks
# -----------------------------------------------------------------------------
def check_books_table(books: pd.DataFrame, id_col: str) -> Dict[str, Any]:
    req_cols = [id_col, "shelves", "shelves_canon", "shelves_canon_content",
                "shelves_noncontent_flags", "n_shelves_raw", "n_shelves_canon", "n_shelves_content"]
    missing = [c for c in req_cols if c not in books.columns]
    issues = {}

    if missing:
        issues["missing_columns"] = missing
        logger.error("Books table missing columns: %s", missing)

    # Dtypes & list columns
    list_cols = ["shelves", "shelves_canon", "shelves_canon_content", "shelves_noncontent_flags"]
    for c in list_cols:
        if c in books.columns:
            bad = books[~books[c].map(is_iterableish)]
            if not bad.empty:
                issues.setdefault("non_list_rows", {})[c] = int(bad.shape[0])
                logger.warning("Column %s has %d rows that are not iterable", c, bad.shape[0])

    # Nulls
    null_ids = books[books[id_col].isna()]
    if not null_ids.empty:
        issues["null_ids"] = int(null_ids.shape[0])

    # Length consistency (flags same length as canon)
    if all(c in books.columns for c in ["shelves_canon", "shelves_noncontent_flags"]):
        lens = (books["shelves_canon"].map(safe_len) - books["shelves_noncontent_flags"].map(safe_len)).abs()
        mism = books[lens != 0]
        if not mism.empty:
            issues["flag_len_mismatch_rows"] = int(mism.shape[0])

    # Count columns consistent
    for a, b in [("n_shelves_raw", "shelves"), ("n_shelves_canon", "shelves_canon"), ("n_shelves_content", "shelves_canon_content")]:
        if a in books.columns and b in books.columns:
            diff = (books[a] - books[b].map(safe_len)).abs()
            bad = (diff > 0).sum()
            if bad:
                issues.setdefault("count_inconsistency", {})[a] = int(bad)

    # Extremes
    stats = {}
    for c in ["n_shelves_raw", "n_shelves_canon", "n_shelves_content"]:
        if c in books.columns:
            s = pd.Series(books[c], dtype="float")
            stats[c] = {
                "min": float(np.nanmin(s.to_numpy())) if len(s) else None,
                "p50": float(np.nanpercentile(s.to_numpy(), 50)) if len(s) else None,
                "p90": float(np.nanpercentile(s.to_numpy(), 90)) if len(s) else None,
                "max": float(np.nanmax(s.to_numpy())) if len(s) else None,
            }
    return {"issues": issues, "stats": stats, "n_rows": int(books.shape[0])}

def check_long_table(df: pd.DataFrame, id_col: str, value_col: str, name: str) -> Dict[str, Any]:
    issues = {}
    req = [id_col, "row_index", value_col]
    missing = [c for c in req if c not in df.columns]
    if missing:
        issues["missing_columns"] = missing

    # Nulls & empties
    null_val = df[df[value_col].isna()]
    empty_val = df[df[value_col].astype(str).str.strip().eq("")]
    if not null_val.empty: issues["null_values"] = int(null_val.shape[0])
    if not empty_val.empty: issues["empty_values"] = int(empty_val.shape[0])

    # Unicode weirdness: non-letters-heavy tokens and control chars
    sample = df[value_col].astype(str).head(500)
    unicode_flags = {
        "has_control_chars": any(any(ud.category(ch).startswith("C") and ch not in ("\t","\n") for ch in s) for s in sample),
        "suspicious_punct_ratio_over_0_5": int(np.mean([sum(1 for ch in s if ud.category(ch).startswith("P"))/max(len(s),1) > 0.5 for s in sample]) * 100)
    }
    issues["unicode_flags_sample"] = unicode_flags

    # Duplicates of (id,row_index,value) indicate upstream duplication
    if all(c in df.columns for c in [id_col, "row_index", value_col]):
        dup = df.duplicated(subset=[id_col, "row_index", value_col]).sum()
        if dup:
            issues["exact_dupe_triplets"] = int(dup)

    return {"name": name, "n_rows": int(df.shape[0]), "issues": issues}

def check_segments(segments_long: pd.DataFrame, id_col: str) -> Dict[str, Any]:
    issues = {}
    req = [id_col, "row_index", "value", "segment", "seg_accepted"]
    missing = [c for c in req if c not in segments_long.columns]
    if missing:
        issues["missing_columns"] = missing
        return {"issues": issues, "n_rows": int(segments_long.shape[0])}

    null_seg = segments_long[segments_long["segment"].isna() | segments_long["segment"].astype(str).str.strip().eq("")]
    if not null_seg.empty:
        issues["null_or_empty_segments"] = int(null_seg.shape[0])

    # Accepted rate
    acc = segments_long["seg_accepted"].fillna(False).astype(bool)
    accepted_rate = float(acc.mean()) if len(acc) else 0.0

    # Very long/very short segments
    seglen = segments_long["segment"].astype(str).str.len()
    too_short = int((seglen == 1).sum())
    too_long = int((seglen > 40).sum())

    # Token quality: many digits or punctuation?
    def bad_token(s: str) -> bool:
        s = str(s)
        digits = sum(ch.isdigit() for ch in s)
        punct = sum(ud.category(ch).startswith("P") for ch in s)
        return digits/ max(len(s),1) > 0.5 or punct / max(len(s),1) > 0.5

    bad_tokens = int(segments_long["segment"].astype(str).map(bad_token).sum())

    return {
        "issues": {**issues, "too_short_segments": too_short, "too_long_segments": too_long, "bad_token_like_segments": bad_tokens},
        "accepted_rate": accepted_rate,
        "n_rows": int(segments_long.shape[0]),
    }

def coverage_vs_canon(canonical_csv: Path, canon_long: pd.DataFrame) -> Dict[str, Any]:
    try:
        canon_map = pd.read_csv(canonical_csv, dtype=str).fillna("")
    except Exception as e:
        logger.error("Failed to read canonical csv: %s", e)
        return {"error": str(e)}

    if not {"shelf_raw", "shelf_canon"}.issubset(canon_map.columns):
        return {"error": "shelf_canonical.csv must have shelf_raw,shelf_canon"}

    # What % of canon_long values are present in shelf_canonical.shelf_canon?
    canon_set = set(canon_map["shelf_canon"].map(norm_key))
    in_output = set(canon_long["shelf_canon"].astype(str).map(norm_key).unique())
    missing_from_map = sorted([x for x in in_output if x not in canon_set])

    return {
        "canon_values_in_output": len(in_output),
        "canon_values_in_map": len(canon_set),
        "canon_values_missing_in_map": len(missing_from_map),
        "sample_missing": missing_from_map[:50],
    }

def noncontent_alignment(noncontent_csv: Path, books: pd.DataFrame) -> Dict[str, Any]:
    try:
        nc = pd.read_csv(noncontent_csv, dtype=str).fillna("")
    except Exception as e:
        logger.error("Failed to read noncontent csv: %s", e)
        return {"error": str(e)}

    col = "shelf"
    if col not in nc.columns:
        for c in ("shelf_canon", "shelf_raw"):
            if c in nc.columns:
                col = c
                break
    nc_keys = set(nc[col].astype(str).map(norm_key))

    # flatten canon & flags to pairs
    pairs = []
    for canon, flags in zip(books.get("shelves_canon", []), books.get("shelves_noncontent_flags", [])):
        if not is_iterableish(canon) or not is_iterableish(flags):
            continue
        for c, f in zip(canon, flags):
            pairs.append((norm_key(str(c)), bool(f)))
    pairs_df = pd.DataFrame(pairs, columns=["canon_key", "flag"])

    if pairs_df.empty:
        return {"pairs_rows": 0, "note": "no canon/flag pairs found"}

    # FP: flagged but not listed in noncontent
    fp = pairs_df[(pairs_df["flag"]) & (~pairs_df["canon_key"].isin(nc_keys))]
    # FN: not flagged but listed in noncontent
    fn = pairs_df[(~pairs_df["flag"]) & (pairs_df["canon_key"].isin(nc_keys))]

    return {
        "pairs_rows": int(pairs_df.shape[0]),
        "false_positives": int(fp.shape[0]),
        "false_negatives": int(fn.shape[0]),
        "fp_rate_pct": float(100 * len(fp) / max(len(pairs_df),1)),
        "fn_rate_pct": float(100 * len(fn) / max(len(pairs_df),1)),
    }

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Deep diagnostics over bridge outputs")
    ap.add_argument("--books", required=True, type=Path)
    ap.add_argument("--raw-long", required=True, type=Path)
    ap.add_argument("--canon-long", required=True, type=Path)
    ap.add_argument("--segments-long", required=True, type=Path)
    ap.add_argument("--canonical-csv", required=True, type=Path)
    ap.add_argument("--segments-csv", required=True, type=Path)
    ap.add_argument("--noncontent-csv", required=True, type=Path)
    ap.add_argument("--alias-csv", required=False, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--id-col", default="work_id")
    ap.add_argument("--print-examples", action="store_true", help="Emit sample CSVs of suspect rows")
    ap.add_argument("--verbose-logs", action="store_true")
    args = ap.parse_args()

    if args.verbose_logs:
        logger.setLevel(logging.DEBUG)

    ensure_outdir(args.out)

    logger.info("Loading inputs…")
    books = read_parquet(args.books)
    raw_long = read_parquet(args.raw_long)
    canon_long = read_parquet(args.canon_long)
    segments_long = read_parquet(args.segments_long)

    # --- Core checks
    logger.info("Running book-table checks…")
    book_res = check_books_table(books, id_col=args.id_col)

    logger.info("Running long-table checks…")
    raw_res = check_long_table(raw_long, args.id_col, "shelf_raw", "shelves_raw_long")
    canon_res = check_long_table(canon_long, args.id_col, "shelf_canon", "shelves_canon_long")
    seg_res = check_segments(segments_long, args.id_col)

    logger.info("Checking coverage vs canonical map…")
    cov_res = coverage_vs_canon(args.canonical_csv, canon_long)

    logger.info("Checking non-content flag alignment…")
    nc_res = noncontent_alignment(args.noncontent_csv, books)

    # Optional alias csv surface
    alias_info = None
    if args.alias_csv and args.alias_csv.exists():
        alias_df = pd.read_csv(args.alias_csv, dtype=str).fillna("")
        alias_info = {
            "n_rows": int(alias_df.shape[0]),
            "has_scores": all(c in alias_df.columns for c in ("jw","edit","jaccard")),
        }

    # --- Summarize
    summary = {
        "timestamp_utc": utcnow_str(),
        "inputs": {
            "books": str(args.books),
            "raw_long": str(args.raw_long),
            "canon_long": str(args.canon_long),
            "segments_long": str(args.segments_long),
            "canonical_csv": str(args.canonical_csv),
            "segments_csv": str(args.segments_csv),
            "noncontent_csv": str(args.noncontent_csv),
            "alias_csv": str(args.alias_csv) if args.alias_csv else None,
        },
        "book_table": book_res,
        "raw_long": raw_res,
        "canon_long": canon_res,
        "segments_long": seg_res,
        "coverage_vs_canon": cov_res,
        "noncontent_alignment": nc_res,
        "alias_info": alias_info,
    }

    # --- Human-readable report
    report_lines = []
    def add(title: str, obj: Any):
        report_lines.append(f"\n== {title} ==\n{json.dumps(obj, indent=2, ensure_ascii=False)}\n")

    add("BOOK_TABLE", book_res)
    add("RAW_LONG", raw_res)
    add("CANON_LONG", canon_res)
    add("SEGMENTS_LONG", seg_res)
    add("COVERAGE_VS_CANON", cov_res)
    add("NONCONTENT_ALIGNMENT", nc_res)
    if alias_info: add("ALIAS_INFO", alias_info)

    # --- Emit examples if flagged
    if args.print_examples:
        logger.info("Writing suspect sample CSVs…")
        # Empty/Null values
        empty_raw = raw_long[raw_long["shelf_raw"].astype(str).str.strip().eq("")]
        if not empty_raw.empty:
            to_csv_sample(empty_raw, args.out / "suspect_examples_empty_raw.csv")
        empty_canon = canon_long[canon_long["shelf_canon"].astype(str).str.strip().eq("")]
        if not empty_canon.empty:
            to_csv_sample(empty_canon, args.out / "suspect_examples_empty_canon.csv")
        empty_seg = segments_long[segments_long["segment"].astype(str).str.strip().eq("")]
        if not empty_seg.empty:
            to_csv_sample(empty_seg, args.out / "suspect_examples_empty_segment.csv")

        # Very long tokens
        long_canon = canon_long[canon_long["shelf_canon"].astype(str).str.len() > 60]
        if not long_canon.empty:
            to_csv_sample(long_canon, args.out / "suspect_examples_very_long_canon.csv")

        # Duplicate triplets
        dup_raw = raw_long[raw_long.duplicated(subset=[args.id_col,"row_index","shelf_raw"], keep=False)]
        if not dup_raw.empty:
            to_csv_sample(dup_raw, args.out / "suspect_examples_dup_triplets_raw.csv")

    # --- Persist
    (args.out / "diagnostics_report.txt").write_text("".join(report_lines), encoding="utf-8")
    with open(args.out / "diagnostics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Diagnostics complete. Report: %s", args.out / "diagnostics_report.txt")

if __name__ == "__main__":
    main()
