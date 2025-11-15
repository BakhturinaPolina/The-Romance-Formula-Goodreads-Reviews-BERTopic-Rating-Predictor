#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bridge Script: Step 1 → Step 2 Integration

Bridges Step 1 (audit + parsed shelves) and Step 2 (shelf normalization)
into an integrated dataset ready for clustering/modeling.

Inputs:
  - parsed books parquet (from 01_parse_lists.py)
  - shelf_canonical.csv (raw->canon)
  - shelf_segments.csv (segmentation result per canon shelf)
  - noncontent_shelves.csv (leakage filter results)
  - [optional] shelf_alias_candidates.csv (for QC)

Outputs (in --out dir):
  - books_with_shelf_norm.parquet
  - shelves_raw_long.parquet
  - shelves_canon_long.parquet
  - segments_long.parquet
  - bridge_summary.json
  - shelf_norm_bridge_log.jsonl (provenance)

Usage:
  python bridge_audit_normalize.py \
    --parsed parse_outputs_full/parsed_books_YYYYmmdd_HHMMSS.parquet \
    --canon normalize_outputs/shelf_canonical.csv \
    --segments normalize_outputs/shelf_segments.csv \
    --noncontent normalize_outputs/noncontent_shelves.csv \
    --alias normalize_outputs/shelf_alias_candidates.csv \
    --out bridge_outputs/
"""

from __future__ import annotations
import argparse
import json
import logging
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Set, Any

import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Logging (quiet by default; use --verbose-logs to increase)
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def utcnow_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception as e:
        logger.warning("pyarrow read failed (%s), retrying with fastparquet...", e)
        return pd.read_parquet(path, engine="fastparquet")

def to_parquet(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False, engine="pyarrow")
    except Exception as e:
        logger.warning("pyarrow write failed (%s), retrying with fastparquet...", e)
        df.to_parquet(path, index=False, engine="fastparquet")

def safe_list(x: Any) -> List:
    """Return x as list if it looks iterable-ish, else []"""
    if isinstance(x, list):
        return x
    if isinstance(x, (np.ndarray, pd.Series, pd.Index)):
        return list(x)
    return []


# -----------------------------------------------------------------------------
# Loaders for Step-2 artifacts
# -----------------------------------------------------------------------------
def load_canonical_map(canon_csv: Path) -> Dict[str, str]:
    """
    Expect columns: shelf_raw, shelf_canon (case-insensitive match on raw).
    """
    df = pd.read_csv(canon_csv, dtype=str).fillna("")
    if not {"shelf_raw", "shelf_canon"}.issubset(set(df.columns)):
        raise ValueError("shelf_canonical.csv must have columns: shelf_raw, shelf_canon")
    # normalize key for robust lookup (case/space)
    def norm_key(s: str) -> str:
        return " ".join(str(s).strip().casefold().split())
    mapping = {norm_key(r): str(c).strip() for r, c in zip(df["shelf_raw"], df["shelf_canon"])}
    return mapping

def load_noncontent(noncontent_csv: Path) -> Set[str]:
    """
    Expect columns: shelf (canonical or raw as exported by Step 2).
    We treat entries casefold+trim as keys.
    """
    df = pd.read_csv(noncontent_csv, dtype=str).fillna("")
    col = None
    for c in ("shelf", "shelf_canon", "shelf_raw"):
        if c in df.columns:
            col = c
            break
    if col is None:
        raise ValueError("noncontent_shelves.csv must contain one of: shelf, shelf_canon, shelf_raw")
    vals = set(" ".join(str(s).strip().casefold().split()) for s in df[col].tolist())
    return vals

def load_segments(segments_csv: Path) -> pd.DataFrame:
    """
    Expect columns at least: shelf_canon, segments, accepted
    Where 'segments' is a JSON-ish list string.
    """
    df = pd.read_csv(segments_csv, dtype=str).fillna("")
    required = {"shelf_canon", "segments", "accepted"}
    if not required.issubset(df.columns):
        raise ValueError(f"shelf_segments.csv must have columns: {required}")
    # normalize canon key
    df["canon_key"] = df["shelf_canon"].map(lambda s: " ".join(str(s).strip().casefold().split()))
    # parse segments safely
    def parse_segs(s: str) -> List[str]:
        s = str(s).strip()
        if not s:
            return []
        # attempt JSON first
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [str(t) for t in obj]
        except Exception:
            pass
        # fallback: split by space/comma
        toks = [t for t in s.replace(",", " ").split() if t]
        return toks
    df["segments_list"] = df["segments"].map(parse_segs)
    df["accepted_bool"] = df["accepted"].str.lower().isin({"1", "true", "yes"})
    return df[["shelf_canon", "canon_key", "segments_list", "accepted_bool"]].drop_duplicates("canon_key")


# -----------------------------------------------------------------------------
# Core bridge logic
# -----------------------------------------------------------------------------
def canonicalize_shelves(
    shelves_raw: List[str],
    canon_map: Dict[str, str],
) -> Tuple[List[str], List[str]]:
    """
    Map raw shelves to canonical; also return normalized keys used for lookup.
    """
    def norm_key(s: str) -> str:
        return " ".join(str(s).strip().casefold().split())

    canon = []
    keys = []
    for raw in shelves_raw:
        k = norm_key(raw)
        keys.append(k)
        c = canon_map.get(k)
        if c is None or str(c).strip() == "":
            # fallback: use normalized key as canon
            c = k
        canon.append(c)
    return canon, keys


def attach_noncontent_flags(
    shelves_canon: List[str],
    noncontent_keys: Set[str],
) -> List[bool]:
    """
    Return a boolean list aligned with shelves_canon marking non-content shelves.
    """
    def norm_key(s: str) -> str:
        return " ".join(str(s).strip().casefold().split())
    return [norm_key(s) in noncontent_keys for s in shelves_canon]


def explode_long(df: pd.DataFrame, col: str, id_col: str = "work_id") -> pd.DataFrame:
    """
    Explode list column into long format with one row per element.
    """
    tmp = df[[id_col, col]].explode(col, ignore_index=False).reset_index()
    tmp.rename(columns={col: "value", "index": "row_index"}, inplace=True)
    return tmp[[id_col, "row_index", "value"]]


def build_segments_long(
    canon_long: pd.DataFrame,
    seg_df: pd.DataFrame,
    id_col: str = "work_id",
) -> pd.DataFrame:
    """
    Join canonical shelves (long) to segments (one-to-many) and explode segments.
    """
    canon_long = canon_long.copy()
    canon_long["canon_key"] = canon_long["value"].map(lambda s: " ".join(str(s).strip().casefold().split()))
    seg = seg_df[["canon_key", "segments_list", "accepted_bool"]].copy()

    merged = canon_long.merge(seg, on="canon_key", how="left")
    # explode segments_list
    merged = merged.explode("segments_list", ignore_index=False)
    merged.rename(columns={"segments_list": "segment", "accepted_bool": "seg_accepted"}, inplace=True)
    merged["segment"] = merged["segment"].fillna("").astype(str)
    merged = merged[merged["segment"].str.len() > 0]
    return merged[[id_col, "row_index", "value", "segment", "seg_accepted"]]


def summarize_bridge(
    df_books: pd.DataFrame,
    shelves_col_raw: str = "shelves",
    shelves_col_canon: str = "shelves_canon",
    shelves_col_content: str = "shelves_canon_content",
) -> Dict[str, Any]:
    n_books = len(df_books)
    n_shelves_raw = int(df_books[shelves_col_raw].map(len).sum()) if shelves_col_raw in df_books.columns else None
    n_shelves_canon = int(df_books[shelves_col_canon].map(len).sum())
    n_shelves_content = int(df_books[shelves_col_content].map(len).sum())

    uniq_canon = len({s for L in df_books[shelves_col_canon] for s in L})
    uniq_content = len({s for L in df_books[shelves_col_content] for s in L})

    return {
        "timestamp_utc": utcnow_str(),
        "n_books": n_books,
        "total_tags_raw": n_shelves_raw,
        "total_tags_canon": n_shelves_canon,
        "total_tags_content": n_shelves_content,
        "unique_canon": uniq_canon,
        "unique_content": uniq_content,
        "avg_tags_raw_per_book": (n_shelves_raw / n_books) if n_shelves_raw is not None else None,
        "avg_tags_canon_per_book": n_shelves_canon / n_books,
        "avg_tags_content_per_book": n_shelves_content / n_books,
    }


def main():
    ap = argparse.ArgumentParser(description="Bridge Step-1 parsed data with Step-2 normalization artifacts.")
    ap.add_argument("--parsed", required=True, type=Path, help="Parquet from 01_parse_lists.py (parsed_books*.parquet)")
    ap.add_argument("--canon", required=True, type=Path, help="shelf_canonical.csv")
    ap.add_argument("--segments", required=True, type=Path, help="shelf_segments.csv")
    ap.add_argument("--noncontent", required=True, type=Path, help="noncontent_shelves.csv")
    ap.add_argument("--alias", required=False, type=Path, help="shelf_alias_candidates.csv (optional)")
    ap.add_argument("--out", required=True, type=Path, help="Output directory for bridged artifacts")
    ap.add_argument("--id-col", default="work_id", help="Primary key column (default: work_id)")
    ap.add_argument("--shelves-col", default="shelves", help="Parsed shelves column name (default: shelves)")
    ap.add_argument("--verbose-logs", action="store_true", help="Increase logging verbosity")
    args = ap.parse_args()

    if args.verbose_logs:
        logger.setLevel(logging.DEBUG)

    ensure_outdir(args.out)

    # --- Load inputs
    logger.info("Loading parsed books: %s", args.parsed)
    books = read_parquet(args.parsed)

    if args.id_col not in books.columns:
        raise KeyError(f"ID column '{args.id_col}' not found in parsed books")

    if args.shelves_col not in books.columns:
        raise KeyError(f"Shelves column '{args.shelves_col}' not found in parsed books")

    # normalize list columns
    books = books.copy()
    books[args.shelves_col] = books[args.shelves_col].map(safe_list)

    logger.info("Loading canonical mapping: %s", args.canon)
    canon_map = load_canonical_map(args.canon)

    logger.info("Loading non-content set: %s", args.noncontent)
    noncontent_keys = load_noncontent(args.noncontent)

    logger.info("Loading segments: %s", args.segments)
    seg_df = load_segments(args.segments)

    # --- Apply canonicalization & flags
    logger.info("Applying canonicalization and flags...")
    shelves_canon = []
    shelves_keys = []
    shelves_noncontent_flags = []
    shelves_content = []

    for raw_list in books[args.shelves_col].tolist():
        canon_list, key_list = canonicalize_shelves(raw_list, canon_map)
        flag_list = attach_noncontent_flags(canon_list, noncontent_keys)
        content_list = [c for c, f in zip(canon_list, flag_list) if not f]

        shelves_canon.append(canon_list)
        shelves_keys.append(key_list)
        shelves_noncontent_flags.append(flag_list)
        shelves_content.append(content_list)

    books["shelves_canon"] = shelves_canon
    books["shelves_lookup_keys"] = shelves_keys
    books["shelves_noncontent_flags"] = shelves_noncontent_flags
    books["shelves_canon_content"] = shelves_content
    books["n_shelves_raw"] = books[args.shelves_col].map(len)
    books["n_shelves_canon"] = books["shelves_canon"].map(len)
    books["n_shelves_content"] = books["shelves_canon_content"].map(len)

    # --- Long tables
    logger.info("Building long-form tables...")
    shelves_raw_long = explode_long(books[[args.id_col, args.shelves_col]].rename(columns={args.shelves_col: "shelves"}), "shelves", id_col=args.id_col)
    shelves_raw_long.rename(columns={"value": "shelf_raw"}, inplace=True)

    shelves_canon_long = explode_long(books[[args.id_col, "shelves_canon"]], "shelves_canon", id_col=args.id_col)
    shelves_canon_long.rename(columns={"value": "shelf_canon"}, inplace=True)

    # Attach segments to canonical shelves (and explode segment tokens)
    segments_long = build_segments_long(
        canon_long=shelves_canon_long[[args.id_col, "row_index", "shelf_canon"]].rename(columns={"shelf_canon": "value"}),
        seg_df=seg_df,
        id_col=args.id_col
    )

    # --- Persist outputs
    out_books = args.out / "books_with_shelf_norm.parquet"
    out_raw_long = args.out / "shelves_raw_long.parquet"
    out_canon_long = args.out / "shelves_canon_long.parquet"
    out_segments_long = args.out / "segments_long.parquet"
    out_log = args.out / "shelf_norm_bridge_log.jsonl"
    out_summary = args.out / "bridge_summary.json"

    logger.info("Saving outputs to %s ...", args.out)
    to_parquet(books, out_books)
    to_parquet(shelves_raw_long, out_raw_long)
    to_parquet(shelves_canon_long, out_canon_long)
    to_parquet(segments_long, out_segments_long)

    # --- Optional: alias candidates QC (pass-through summary)
    alias_summary = None
    if args.alias and args.alias.exists():
        try:
            alias_df = pd.read_csv(args.alias, dtype=str).fillna("")
            alias_summary = {
                "n_alias_rows": int(alias_df.shape[0]),
                "avg_jw": float(pd.to_numeric(alias_df.get("jw", pd.Series(dtype=float)), errors="coerce").dropna().mean())
                    if "jw" in alias_df.columns else None
            }
        except Exception as e:
            logger.warning("Failed to summarize alias candidates: %s", e)

    # --- Bridge summary
    summary = summarize_bridge(books, shelves_col_raw=args.shelves_col)
    if alias_summary:
        summary["alias_summary"] = alias_summary

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # --- Provenance log (append)
    provenance = {
        "timestamp_utc": utcnow_str(),
        "parsed_input": str(args.parsed),
        "canon_csv": str(args.canon),
        "segments_csv": str(args.segments),
        "noncontent_csv": str(args.noncontent),
        "alias_csv": str(args.alias) if args.alias else None,
        "outputs": {
            "books_with_shelf_norm": str(out_books),
            "shelves_raw_long": str(out_raw_long),
            "shelves_canon_long": str(out_canon_long),
            "segments_long": str(out_segments_long),
            "bridge_summary": str(out_summary),
        },
        "id_col": args.id_col,
        "shelves_col": args.shelves_col,
        "env": {
            "pwd": os.getcwd(),
            "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
        }
    }
    with open(out_log, "a", encoding="utf-8") as f:
        f.write(json.dumps(provenance, ensure_ascii=False) + "\n")

    logger.info("Bridge complete. Main output: %s", out_books)


if __name__ == "__main__":
    main()
