#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deep-dive diagnostics for Shelf Normalization outputs.

Inputs (produced by Step 2):
  - shelf_canonical.csv            (shelf_raw, shelf_canon, reason)
  - shelf_alias_candidates.csv     (shelf_a, shelf_b, jw, edit, jaccard, decision_hint, ...)
  - shelf_segments.csv             (shelf_canon, segments, accepted, guard1, guard2, evidence_count, ...)
  - noncontent_shelves.csv         (pattern, type, note, ...)
  - shelf_normalization_log.jsonl  (one record per decision)
  - segments_vocab.txt             (one token per line)

Outputs:
  - diagnostics_outputs/
      diagnostics_report.txt
      diagnostics_summary.json
      alias_dryrun_samples.csv                (optional)
      alias_dryrun_top_pairs.csv              (optional)

Usage:
  python diagnostics_explore.py \
    --in-dir ./shelf_outputs_full \
    --out-dir ./diagnostics_outputs \
    --alias-dryrun-size 10000 \
    --alias-topk 200 \
    --verbose-logs
"""

import argparse
import csv
import io
import json
import logging
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional, Set

import pandas as pd

# Optional: RapidFuzz for fast JW/edit; handled gracefully if absent
try:
    from rapidfuzz.distance import JaroWinkler, DamerauLevenshtein
    HAVE_RAPIDFUZZ = True
except Exception:
    HAVE_RAPIDFUZZ = False


# ------------------------- CLI & Logging ------------------------------------- #

def make_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diagnostics explorer for shelf normalization outputs."
    )
    p.add_argument("--in-dir", type=Path, required=True,
                   help="Directory containing Step 2 outputs (CSV/JSONL/TXT).")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Directory for diagnostics artifacts.")
    p.add_argument("--alias-dryrun-size", type=int, default=10000,
                   help="Number of unique shelves to sample for alias dry-run.")
    p.add_argument("--alias-topk", type=int, default=200,
                   help="Top candidate pairs to save from dry-run.")
    p.add_argument("--jw-threshold", type=float, default=0.92,
                   help="Jaro–Winkler threshold used in dry-run.")
    p.add_argument("--jaccard3-threshold", type=float, default=0.80,
                   help="Char-3gram Jaccard threshold used in dry-run.")
    p.add_argument("--edit-threshold", type=int, default=1,
                   help="Damerau–Levenshtein edit threshold used in dry-run.")
    p.add_argument("--verbose-logs", action="store_true",
                   help="Enable DEBUG logs.")
    return p.parse_args()


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence matplotlib etc. unless explicitly requested
    if not verbose:
        for noisy in ["matplotlib", "PIL", "fsspec", "urllib3", "numba", "pyarrow"]:
            logging.getLogger(noisy).setLevel(logging.WARNING)


# ------------------------- Utilities ----------------------------------------- #

def read_csv_safe(path: Path, **kwargs) -> Optional[pd.DataFrame]:
    if not path.exists():
        logging.warning(f"Missing file: {path}")
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        logging.error(f"Failed to read {path}: {e}")
        return None


def read_jsonl_safe(path: Path) -> List[dict]:
    if not path.exists():
        logging.warning(f"Missing file: {path}")
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                logging.warning(f"Bad JSONL line {i} in {path.name}: {e}")
    return out


def jaccard_char_ngrams(a: str, b: str, n: int = 3) -> float:
    def grams(s: str) -> Set[str]:
        s = s.strip()
        if len(s) < n:
            return {s} if s else set()
        return {s[i:i+n] for i in range(len(s)-n+1)}
    A, B = grams(a), grams(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def tokenize(s: str) -> List[str]:
    return [t for t in re.split(r"\s+", s.strip()) if t]


def is_dash_heavy(s: str) -> bool:
    return bool(re.fullmatch(r"[-_]+.*", s)) or (s.count("-") + s.count("_")) >= max(3, len(s)//2)


def safe_eval_segments(field: str) -> List[str]:
    """Segments may be JSON, Python-literal, or comma-joined. Be lenient."""
    if pd.isna(field) or not str(field).strip():
        return []
    text = str(field).strip()
    # Try JSON list
    if text.startswith("[") and text.endswith("]"):
        try:
            arr = json.loads(text)
            return [str(x) for x in arr]
        except Exception:
            pass
    # Try Python literal list
    if text.startswith("[") and text.endswith("]"):
        try:
            import ast
            arr = ast.literal_eval(text)
            return [str(x) for x in arr]
        except Exception:
            pass
    # Fallback: split by comma
    return [t.strip() for t in text.split(",") if t.strip()]


# ------------------------- Diagnostics --------------------------------------- #

@dataclass
class CanonStats:
    n_rows: int = 0
    n_unique_raw: int = 0
    n_unique_canon: int = 0
    compression_ratio: float = 0.0
    n_identity_maps: int = 0
    n_blank_canon: int = 0
    top_unmapped: List[Tuple[str, int]] = None
    dash_heavy_count: int = 0


def analyze_canonicalization(df: pd.DataFrame) -> CanonStats:
    """
    Assumes columns: shelf_raw, shelf_canon, (optional) reason
    """
    logging.info("Analyzing canonicalization coverage...")
    needed = {"shelf_raw", "shelf_canon"}
    missing = needed - set(df.columns)
    if missing:
        logging.warning(f"shelf_canonical.csv missing columns: {missing}")

    # Fill NA to avoid errors
    df = df.copy()
    df["shelf_raw"] = df.get("shelf_raw", "").astype(str)
    df["shelf_canon"] = df.get("shelf_canon", "").astype(str)

    n_rows = len(df)
    uniq_raw = df["shelf_raw"].nunique(dropna=False)
    uniq_canon = df["shelf_canon"].nunique(dropna=False)
    compression = (uniq_canon / uniq_raw) if uniq_raw else 0.0

    identity_mask = (df["shelf_canon"].str.strip() == df["shelf_raw"].str.strip())
    n_identity = int(identity_mask.sum())
    n_blank = int((df["shelf_canon"].str.strip() == "").sum())

    # Unmapped heuristic: identity OR blank canon
    unmapped_mask = identity_mask | (df["shelf_canon"].str.strip() == "")
    unmapped = df.loc[unmapped_mask, "shelf_raw"].value_counts()
    top_unmapped = list(unmapped.head(100).items())

    dash_heavy = int(df["shelf_raw"].apply(is_dash_heavy).sum())

    logging.debug(f"Canonicalization: uniq_raw={uniq_raw}, uniq_canon={uniq_canon}, "
                  f"compression={compression:.4f}, identity={n_identity}, blanks={n_blank}, "
                  f"dash_heavy={dash_heavy}")

    return CanonStats(
        n_rows=n_rows,
        n_unique_raw=uniq_raw,
        n_unique_canon=uniq_canon,
        compression_ratio=round(compression, 6),
        n_identity_maps=n_identity,
        n_blank_canon=n_blank,
        top_unmapped=top_unmapped,
        dash_heavy_count=dash_heavy,
    )


@dataclass
class SegStats:
    n_rows: int = 0
    n_accept: int = 0
    accept_rate: float = 0.0
    len1_2_rejects: int = 0
    digit_or_punct_rejects: int = 0
    top_rejected_tokens: List[Tuple[str, int]] = None
    examples_accept: List[Tuple[str, List[str]]] = None
    examples_reject: List[Tuple[str, List[str]]] = None


def analyze_segmentation(df: pd.DataFrame) -> SegStats:
    """
    Assumes columns: shelf_canon, segments, accepted, guard1, guard2, evidence_count
    """
    logging.info("Analyzing segmentation quality...")
    needed = {"shelf_canon", "segments", "accepted"}
    missing = needed - set(df.columns)
    if missing:
        logging.warning(f"shelf_segments.csv missing columns: {missing}")

    df = df.copy()
    df["shelf_canon"] = df.get("shelf_canon", "").astype(str)
    df["segments_raw"] = df.get("segments", "").astype(str)
    df["accepted"] = df.get("accepted", False).astype(bool)

    # Parse segments robustly
    df["segments_list"] = df["segments_raw"].apply(safe_eval_segments)

    n_rows = len(df)
    n_accept = int(df["accepted"].sum())
    accept_rate = (n_accept / n_rows) if n_rows else 0.0

    # Reject heuristics
    rejected = df.loc[~df["accepted"]].copy()
    # Flatten rejected tokens
    rej_tokens = [tok for segs in rejected["segments_list"] for tok in segs] if len(rejected) else []
    # Heuristic buckets
    len1_2 = [t for t in rej_tokens if len(t) <= 2]
    digit_punct = [t for t in rej_tokens if (t.isdigit() or re.fullmatch(r"[\W_]+", t) is not None)]
    top_rej = Counter(rej_tokens).most_common(50)

    # Examples
    examples_accept = []
    examples_reject = []
    for _, row in df[df["accepted"]].head(20).iterrows():
        examples_accept.append((row["shelf_canon"], row["segments_list"]))
    for _, row in df[~df["accepted"]].head(20).iterrows():
        examples_reject.append((row["shelf_canon"], row["segments_list"]))

    logging.debug(f"Segmentation: n={n_rows}, accept={n_accept} ({accept_rate:.2%}), "
                  f"len<=2 rejects={len(len1_2)}, digit/punct rejects={len(digit_punct)}")

    return SegStats(
        n_rows=n_rows,
        n_accept=n_accept,
        accept_rate=round(accept_rate, 6),
        len1_2_rejects=len(len1_2),
        digit_or_punct_rejects=len(digit_punct),
        top_rejected_tokens=top_rej,
        examples_accept=examples_accept,
        examples_reject=examples_reject,
    )


@dataclass
class AliasStats:
    n_rows: int = 0
    has_candidates: bool = False
    top_by_jw: List[Tuple[str, str, float]] = None
    samples_saved: Optional[Path] = None
    top_pairs_saved: Optional[Path] = None


def alias_dryrun(
    canon_df: pd.DataFrame,
    out_dir: Path,
    sample_size: int,
    jw_thr: float,
    j3_thr: float,
    ed_thr: int,
    topk: int
) -> AliasStats:
    """
    If alias file is empty or missing, we compute candidate pairs on a sample of shelves.
    Blocking: first character or first token match; then JW/Jaccard/ED.
    """
    logging.info("Running alias dry-run (sanity exploration)...")
    rng = random.Random(1337)

    shelves = sorted(set(canon_df.get("shelf_canon", pd.Series([], dtype=str)).astype(str).tolist()))
    if not shelves:
        logging.warning("No shelves available for alias dry-run.")
        return AliasStats(n_rows=0, has_candidates=False)

    sample = shelves if len(shelves) <= sample_size else rng.sample(shelves, sample_size)

    # Save sample for reproducibility
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = out_dir / "alias_dryrun_samples.csv"
    pd.DataFrame({"shelf": sample}).to_csv(sample_path, index=False)

    # Build blocks
    def first_key(s: str) -> str:
        return (tokenize(s)[0] if tokenize(s) else s[:1]).lower()

    blocks = defaultdict(list)
    for s in sample:
        blocks[first_key(s)].append(s)

    logging.debug(f"Dry-run blocks: {len(blocks)} keys")

    candidates = []
    def jw(a,b):
        if HAVE_RAPIDFUZZ:
            return JaroWinkler.similarity(a, b) / 100.0  # RF returns 0..100
        # crude fallback
        return 1.0 - (abs(len(a)-len(b)) / (len(a)+len(b)+1e-9))

    def ed(a,b):
        if HAVE_RAPIDFUZZ:
            return int(DamerauLevenshtein.distance(a, b))
        # crude fallback: Hamming-like when lengths match; else big number
        return sum(ch1 != ch2 for ch1, ch2 in zip(a, b)) + abs(len(a) - len(b))

    for key, items in blocks.items():
        if len(items) < 2:
            continue
        items = sorted(items)
        # Quadratic per block; blocks are small-ish
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                a, b = items[i], items[j]
                # quick length prefilter
                if abs(len(a) - len(b)) > 3:
                    continue
                jw_score = jw(a, b)
                if jw_score < (jw_thr - 0.03):  # early prune
                    continue
                j3 = jaccard_char_ngrams(a, b, n=3)
                edist = ed(a, b)
                # decision hint
                good = (jw_score >= jw_thr and edist <= ed_thr) or (jw_score >= jw_thr and j3 >= j3_thr)
                candidates.append({
                    "shelf_a": a, "shelf_b": b,
                    "jw": round(jw_score, 4),
                    "edit": int(edist),
                    "jaccard3": round(j3, 4),
                    "decision_hint": "suggest" if good else "inspect"
                })

    cand_df = pd.DataFrame(candidates)
    has_cand = len(cand_df) > 0
    top_pairs_path = None

    if has_cand:
        cand_df.sort_values(by=["decision_hint", "jw", "jaccard3"], ascending=[True, False, False], inplace=True)
        top_pairs = cand_df.head(topk)
        top_pairs_path = out_dir / "alias_dryrun_top_pairs.csv"
        top_pairs.to_csv(top_pairs_path, index=False)

    logging.info(f"Alias dry-run found {len(cand_df)} candidate pairs (saved top {min(topk, len(cand_df))}).")

    top_by_jw = []
    if has_cand:
        for _, r in cand_df.head(50).iterrows():
            top_by_jw.append((r["shelf_a"], r["shelf_b"], float(r["jw"])))

    return AliasStats(
        n_rows=len(cand_df),
        has_candidates=has_cand,
        top_by_jw=top_by_jw,
        samples_saved=sample_path,
        top_pairs_saved=top_pairs_path,
    )


# ------------------------- Main ---------------------------------------------- #

def main():
    args = make_cli()
    setup_logging(args.verbose_logs)

    in_dir: Path = args.in_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load artifacts
    canon_path = in_dir / "shelf_canonical.csv"
    alias_path = in_dir / "shelf_alias_candidates.csv"
    seg_path = in_dir / "shelf_segments.csv"
    noncontent_path = in_dir / "noncontent_shelves.csv"
    log_path = in_dir / "shelf_normalization_log.jsonl"
    seg_vocab_path = in_dir / "segments_vocab.txt"

    canon_df = read_csv_safe(canon_path)
    alias_df = read_csv_safe(alias_path)
    seg_df = read_csv_safe(seg_path)
    noncontent_df = read_csv_safe(noncontent_path)
    logs = read_jsonl_safe(log_path)

    # Read vocab (optional)
    segments_vocab = []
    if seg_vocab_path.exists():
        segments_vocab = [line.strip() for line in seg_vocab_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    # -------- Canonicalization analysis -------- #
    canon_stats = None
    if canon_df is not None and len(canon_df):
        canon_stats = analyze_canonicalization(canon_df)
    else:
        logging.warning("Canonicalization file missing or empty; skipping coverage analysis.")

    # -------- Segmentation analysis -------- #
    seg_stats = None
    if seg_df is not None and len(seg_df):
        seg_stats = analyze_segmentation(seg_df)
    else:
        logging.warning("Segmentation file missing or empty; skipping segmentation analysis.")

    # -------- Alias analysis (native or dry-run) -------- #
    alias_stats = AliasStats(n_rows=0, has_candidates=False)
    if alias_df is not None and len(alias_df):
        logging.info(f"Alias candidates present: {len(alias_df):,}")
        alias_stats = AliasStats(
            n_rows=len(alias_df),
            has_candidates=True,
            top_by_jw=[
                (r["shelf_a"], r["shelf_b"], float(r.get("jw", float("nan"))))
                for _, r in alias_df.sort_values(by=["jw"], ascending=False).head(50).iterrows()
            ]
        )
    else:
        if canon_df is not None and len(canon_df):
            alias_stats = alias_dryrun(
                canon_df=canon_df,
                out_dir=out_dir,
                sample_size=args.alias_dryrun_size,
                jw_thr=args.jw_threshold,
                j3_thr=args.jaccard3_threshold,
                ed_thr=args.edit_threshold,
                topk=args.alias_topk
            )

    # -------- Noncontent quick checks -------- #
    noncontent_info = {}
    if noncontent_df is not None and len(noncontent_df):
        noncontent_info = {
            "n_rows": int(len(noncontent_df)),
            "types": dict(noncontent_df.get("type", pd.Series([], dtype=str)).value_counts().head(20))
        }

    # -------- Build TXT report -------- #
    txt_lines = []
    def add(line="", end="\n"):
        txt_lines.append(line + end)

    add("="*70)
    add("Shelf Normalization Diagnostics")
    add("="*70)
    add(f"Timestamp UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"Input dir: {in_dir}")
    add(f"Output dir: {out_dir}")
    add("")

    # Canon
    add("## Canonicalization")
    if canon_stats:
        add(f"- Rows: {canon_stats.n_rows:,}")
        add(f"- Unique shelves (raw): {canon_stats.n_unique_raw:,}")
        add(f"- Unique shelves (canon): {canon_stats.n_unique_canon:,}")
        add(f"- Compression ratio (canon/raw): {canon_stats.compression_ratio:.6f}")
        add(f"- Identity maps (canon == raw): {canon_stats.n_identity_maps:,}")
        add(f"- Blank canon values: {canon_stats.n_blank_canon:,}")
        add(f"- Dash-heavy raw shelves: {canon_stats.dash_heavy_count:,}")
        add("- Top 20 unmapped (identity or blank):")
        for s, c in (canon_stats.top_unmapped or [])[:20]:
            add(f"  • {s}  (n={c})")
    else:
        add("No canonicalization data available.")
    add("")

    # Segmentation
    add("## Segmentation")
    if seg_stats:
        add(f"- Rows: {seg_stats.n_rows:,}")
        add(f"- Accepted: {seg_stats.n_accept:,} ({seg_stats.accept_rate:.2%})")
        add(f"- Reject len<=2 tokens (heuristic): {seg_stats.len1_2_rejects:,}")
        add(f"- Reject digit/punct tokens (heuristic): {seg_stats.digit_or_punct_rejects:,}")
        add("- Top 20 rejected tokens:")
        for t, c in (seg_stats.top_rejected_tokens or [])[:20]:
            add(f"  • {t}  (n={c})")
        add("- Examples (accepted, first 10):")
        for s, segs in (seg_stats.examples_accept or [])[:10]:
            add(f"  ✓ {s} → {segs}")
        add("- Examples (rejected, first 10):")
        for s, segs in (seg_stats.examples_reject or [])[:10]:
            add(f"  ✗ {s} → {segs}")
    else:
        add("No segmentation data available.")
    add("")

    # Alias
    add("## Alias Candidates")
    if alias_stats.has_candidates:
        add(f"- Candidates present: {alias_stats.n_rows:,}")
        add("- Top by JW (first 20):")
        for a, b, jw in (alias_stats.top_by_jw or [])[:20]:
            add(f"  • {a} ↔ {b}  (JW={jw:.3f})")
    else:
        add(f"- No alias candidates present in file. Dry-run produced {alias_stats.n_rows:,} pairs.")
        if alias_stats.samples_saved:
            add(f"  Dry-run samples: {alias_stats.samples_saved}")
        if alias_stats.top_pairs_saved:
            add(f"  Dry-run top pairs: {alias_stats.top_pairs_saved}")
    add("")

    # Noncontent
    add("## Non-content shelves")
    if noncontent_info:
        add(f"- Rows: {noncontent_info.get('n_rows', 0):,}")
        types = noncontent_info.get("types", {})
        if types:
            add("- Types (top):")
            for k, v in list(types.items())[:10]:
                add(f"  • {k}: {v}")
    else:
        add("No non-content list available.")
    add("")

    # Logs
    add("## Normalization logs")
    if logs:
        add(f"- Log lines: {len(logs):,}")
        # quick counts by 'action' field if present
        acts = Counter([rec.get("action", "unknown") for rec in logs])
        add("- Actions (top):")
        for k, v in acts.most_common(10):
            add(f"  • {k}: {v}")
    else:
        add("No normalization logs available.")
    add("")

    # Vocab
    add("## Segments vocabulary")
    if segments_vocab:
        add(f"- Vocabulary size: {len(segments_vocab):,}")
        add("- Sample tokens (first 50):")
        for i, tok in enumerate(segments_vocab[:50]):
            if i % 10 == 0:
                add("  ", end="")
            add(f"{tok:<12}", end="")
            if (i + 1) % 10 == 0:
                add("")
        if len(segments_vocab) > 50:
            add(f"\n  ... and {len(segments_vocab) - 50:,} more")
    else:
        add("No segments vocabulary available.")
    add("")

    # -------- Write outputs -------- #
    report_path = out_dir / "diagnostics_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))

    # JSON summary
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "canonicalization": {
            "n_rows": canon_stats.n_rows if canon_stats else 0,
            "n_unique_raw": canon_stats.n_unique_raw if canon_stats else 0,
            "n_unique_canon": canon_stats.n_unique_canon if canon_stats else 0,
            "compression_ratio": canon_stats.compression_ratio if canon_stats else 0.0,
            "n_identity_maps": canon_stats.n_identity_maps if canon_stats else 0,
            "n_blank_canon": canon_stats.n_blank_canon if canon_stats else 0,
            "dash_heavy_count": canon_stats.dash_heavy_count if canon_stats else 0,
        },
        "segmentation": {
            "n_rows": seg_stats.n_rows if seg_stats else 0,
            "n_accept": seg_stats.n_accept if seg_stats else 0,
            "accept_rate": seg_stats.accept_rate if seg_stats else 0.0,
            "len1_2_rejects": seg_stats.len1_2_rejects if seg_stats else 0,
            "digit_or_punct_rejects": seg_stats.digit_or_punct_rejects if seg_stats else 0,
        },
        "alias_candidates": {
            "n_rows": alias_stats.n_rows,
            "has_candidates": alias_stats.has_candidates,
            "samples_saved": str(alias_stats.samples_saved) if alias_stats.samples_saved else None,
            "top_pairs_saved": str(alias_stats.top_pairs_saved) if alias_stats.top_pairs_saved else None,
        },
        "noncontent": noncontent_info,
        "logs": {
            "n_lines": len(logs),
            "actions": dict(Counter([rec.get("action", "unknown") for rec in logs]).most_common(10))
        },
        "vocabulary": {
            "n_tokens": len(segments_vocab),
        },
        "rapidfuzz_available": HAVE_RAPIDFUZZ,
    }

    summary_path = out_dir / "diagnostics_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logging.info(f"Diagnostics complete. Report: {report_path}")
    logging.info(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
