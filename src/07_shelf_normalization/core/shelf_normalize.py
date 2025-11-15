#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2: Shelf normalization pipeline ("from messy tags to umbrella shelves")
- Canonicalization (2.1): deterministic normalization + alias suggestions
- Segmentation (2.2): DFKI-conservative CamelCase / concatenation split
- Deliverables (2.3): canonical tables, alias candidates, segments, logs

Run:
  python shelf_normalize.py --from-audit audit_outputs_full \
      --parsed-parquet parse_outputs_full/parsed_books_*.parquet \
      --outdir shelf_norm_outputs \
      --n-top-shelves 400000 --jw-threshold 0.94 --j3-threshold 0.80 \
      --min-seg-evidence 5 --zipf-min 3.0
"""
from __future__ import annotations

import argparse, json, logging, math, os, re, sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Set, Optional, Any, Counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import unicodedata
from collections import Counter, defaultdict

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Try to import optional dependencies with fallbacks
try:
    from rapidfuzz.distance import JaroWinkler, DamerauLevenshtein
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    logging.warning("rapidfuzz not available. Install with: pip install rapidfuzz")

try:
    from wordfreq import zipf_frequency
    WORDFREQ_AVAILABLE = True
except ImportError:
    WORDFREQ_AVAILABLE = False
    logging.warning("wordfreq not available. Install with: pip install wordfreq")

# ---------- logging ----------
LOG = logging.getLogger("shelf_norm")

def set_logging(quiet: bool = True):
    lvl = logging.INFO if not quiet else logging.WARNING
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )

@dataclass
class AuditContext:
    audit_dir: Path
    audit_json: Path
    overdisp_json: Path
    parsed_parquet: Optional[Path]
    git_hash: Optional[str]
    timestamp: Optional[str]

def load_audit_context(audit_dir: Path,
                       parsed_parquet: Optional[Path]) -> AuditContext:
    """
    Load audit context from directory containing audit_results.json and overdispersion_tests.json.
    """
    audit_json = audit_dir / "audit_results.json"
    overdisp_json = audit_dir / "overdispersion_tests.json"
    
    if not audit_json.exists():
        raise FileNotFoundError(f"audit_results.json not found in {audit_dir}")
    if not overdisp_json.exists():
        raise FileNotFoundError(f"overdispersion_tests.json not found in {audit_dir}")
    
    # Try to get git hash
    git_hash = None
    try:
        import subprocess
        result = subprocess.run(["git", "rev-parse", "HEAD"], 
                              capture_output=True, text=True, cwd=project_root)
        if result.returncode == 0:
            git_hash = result.stdout.strip()
    except Exception:
        pass
    
    # If parsed_parquet not provided, try to find it in audit_results.json
    if parsed_parquet is None:
        try:
            with open(audit_json, 'r') as f:
                audit_data = json.load(f)
                parsed_hint = audit_data.get('parsed_parquet_path')
                if parsed_hint and Path(parsed_hint).exists():
                    parsed_parquet = Path(parsed_hint)
                    LOG.info(f"Found parsed parquet from audit context: {parsed_parquet}")
        except Exception as e:
            LOG.warning(f"Could not extract parsed parquet from audit context: {e}")
    
    return AuditContext(
        audit_dir=audit_dir,
        audit_json=audit_json,
        overdisp_json=overdisp_json,
        parsed_parquet=parsed_parquet,
        git_hash=git_hash,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

def is_noncontent(shelf: str) -> Optional[str]:
    """
    Check if shelf matches non-content patterns.
    Returns category name if non-content, None if content.
    """
    if not shelf or not isinstance(shelf, str):
        return "generic_noncontent"
    
    s = shelf.strip().lower()
    
    # Empty or very short
    if len(s) <= 1:
        return "generic_noncontent"
    
    # Pure numbers
    if s.isdigit():
        return "ratings_valence"
    
    # Star ratings
    if re.match(r'^\d+[\*★☆]+$', s) or re.match(r'^[\*★☆]+\d*$', s):
        return "ratings_valence"
    
    # Reading status
    status_patterns = [
        r'to.?read', r'want.?to.?read', r'wtr', r'wishlist',
        r'currently.?reading', r'cr', r'reading.?now',
        r'read', r'finished', r'done',
        r'dnf', r'did.?not.?finish', r'abandoned'
    ]
    for pattern in status_patterns:
        if re.search(pattern, s):
            return "process_status"
    
    # Years and challenges
    if re.match(r'^\d{4}$', s) or re.search(r'\d{4}', s):
        return "dates_campaigns"
    
    # Formats
    format_patterns = [
        r'hardcover', r'hc', r'paperback', r'pb', r'ebook', r'kindle',
        r'audiobook', r'audio', r'pdf', r'epub'
    ]
    for pattern in format_patterns:
        if re.search(pattern, s):
            return "format_edition"
    
    # Sources
    source_patterns = [
        r'amazon', r'library', r'borrowed', r'owned', r'purchased',
        r'gift', r'free', r'arc', r'netgalley'
    ]
    for pattern in source_patterns:
        if re.search(pattern, s):
            return "source_acquisition"
    
    # Personal organization
    org_patterns = [
        r'favorites?', r'favs?', r'best', r'top', r'worst',
        r'owned', r'collection', r'shelf', r'bookshelf'
    ]
    for pattern in org_patterns:
        if re.search(pattern, s):
            return "personal_org"
    
    return None

def canon_key(raw: str) -> str:
    # normalize → casefold → collapse whitespace
    if not raw or not isinstance(raw, str):
        return ""
    
    # Unicode normalization
    s = unicodedata.normalize('NFKC', raw)
    
    # Replace separators with spaces
    s = re.sub(r'[-_]+', ' ', s)
    
    # Remove edge punctuation
    s = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', s)
    
    # Collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    
    # Remove diacritics
    s = ''.join(c for c in unicodedata.normalize('NFD', s) 
                if unicodedata.category(c) != 'Mn')
    
    # Case fold
    s = s.casefold()
    
    return s

def camel_split(token: str) -> List[str]:
    # Split on boundaries between lower->Upper or digit->Upper; keep order
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", token)
    return parts if parts else [token]

def zipf_ok(w: str, zipf_min: float) -> bool:
    if not WORDFREQ_AVAILABLE:
        return True  # Skip if wordfreq not available
    try:
        freq = zipf_frequency(w, 'en')
        return freq >= zipf_min
    except Exception:
        return True

def lexicon_ok(parts: List[str], zipf_min: float, domain_lex: Set[str]) -> bool:
    if not parts:
        return False
    # At least one part should be in domain lexicon OR pass zipf threshold
    for part in parts:
        if part.lower() in domain_lex or zipf_ok(part, zipf_min):
            return True
    return False

def jaccard_char_ngrams(a: str, b: str, n: int = 3) -> float:
    def ngrams(s: str) -> Set[str]:
        s = s.strip()
        if len(s) < n:
            return {s} if s else set()
        return {s[i:i+n] for i in range(len(s)-n+1)}
    
    A, B = ngrams(a), ngrams(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def simple_edit_distance(a: str, b: str) -> int:
    """Simple edit distance implementation."""
    if len(a) < len(b):
        return simple_edit_distance(b, a)
    
    if len(b) == 0:
        return len(a)
    
    previous_row = list(range(len(b) + 1))
    for i, c1 in enumerate(a):
        current_row = [i + 1]
        for j, c2 in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def jaro_winkler_similarity(a: str, b: str) -> float:
    """Jaro-Winkler similarity implementation."""
    if a == b:
        return 1.0
    
    len_a, len_b = len(a), len(b)
    if len_a == 0 or len_b == 0:
        return 0.0
    
    match_window = max(len_a, len_b) // 2 - 1
    if match_window < 0:
        match_window = 0
    
    a_matches = [False] * len_a
    b_matches = [False] * len_b
    
    matches = 0
    transpositions = 0
    
    # Find matches
    for i in range(len_a):
        start = max(0, i - match_window)
        end = min(i + match_window + 1, len_b)
        for j in range(start, end):
            if b_matches[j] or a[i] != b[j]:
                continue
            a_matches[i] = True
            b_matches[j] = True
            matches += 1
            break
    
    if matches == 0:
        return 0.0
    
    # Count transpositions
    k = 0
    for i in range(len_a):
        if not a_matches[i]:
            continue
        while not b_matches[k]:
            k += 1
        if a[i] != b[k]:
            transpositions += 1
        k += 1
    
    jaro = (matches / len_a + matches / len_b + (matches - transpositions / 2) / matches) / 3
    
    # Winkler prefix bonus
    prefix = 0
    for i in range(min(len_a, len_b, 4)):
        if a[i] == b[i]:
            prefix += 1
        else:
            break
    
    return jaro + (0.1 * prefix * (1 - jaro))

def damerau_levenshtein_distance(a: str, b: str) -> int:
    """Damerau-Levenshtein distance implementation."""
    if RAPIDFUZZ_AVAILABLE:
        return DamerauLevenshtein.distance(a, b)
    else:
        return simple_edit_distance(a, b)

def stream_unique_shelves(parquet_path: Path,
                          col: str = "shelves",
                          nrows: Optional[int] = None) -> Counter:
    """
    Stream through parquet and count unique shelves efficiently.
    """
    LOG.info(f"Streaming shelves from {parquet_path}")
    
    try:
        parquet_file = pq.ParquetFile(parquet_path)
    except Exception as e:
        LOG.error(f"Failed to open parquet file: {e}")
        raise
    
    shelf_counts = Counter()
    total_rows = 0
    
    for batch in parquet_file.iter_batches(batch_size=10000, columns=[col]):
        if nrows and total_rows >= nrows:
            break
            
        batch_df = batch.to_pandas()
        total_rows += len(batch_df)
        
        for shelves_list in batch_df[col]:
            if isinstance(shelves_list, list):
                for shelf in shelves_list:
                    if shelf and isinstance(shelf, str):
                        shelf_counts[shelf] += 1
    
    LOG.info(f"Processed {total_rows:,} rows, found {len(shelf_counts):,} unique shelves")
    return shelf_counts

def run_pipeline(ctx: AuditContext,
                 outdir: Path,
                 jw_thr: float,
                 j3_thr: float,
                 ed_max: int,
                 min_seg_evidence: int,
                 zipf_min: float,
                 top_k: Optional[int],
                 verbose_logs: bool):
    """
    Main pipeline execution.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Load parsed data
    if ctx.parsed_parquet is None:
        raise ValueError("No parsed parquet file available")
    
    LOG.info(f"Loading parsed data from {ctx.parsed_parquet}")
    df = pd.read_parquet(ctx.parsed_parquet)
    
    if "shelves" not in df.columns:
        raise ValueError("shelves column not found in parsed data")
    
    # Stream unique shelves
    shelf_counts = stream_unique_shelves(ctx.parsed_parquet, nrows=top_k)
    
    # Limit to top-k if specified
    if top_k:
        shelf_counts = Counter(dict(shelf_counts.most_common(top_k)))
        LOG.info(f"Limited to top {top_k:,} shelves")
    
    # Canonicalization
    LOG.info("Starting canonicalization...")
    canon_map = {}
    noncontent_shelves = []
    
    for shelf, count in shelf_counts.items():
        canon = canon_key(shelf)
        canon_map[shelf] = canon
        
        # Check for non-content
        nc_category = is_noncontent(shelf)
        if nc_category:
            noncontent_shelves.append({
                'shelf_raw': shelf,
                'shelf_canon': canon,
                'category': nc_category,
                'count': count
            })
    
    # Save canonical mapping
    canon_df = pd.DataFrame([
        {
            'shelf_raw': raw,
            'shelf_canon': canon,
            'reason': 'normalized',
            'count': shelf_counts[raw]
        }
        for raw, canon in canon_map.items()
    ])
    
    out_canon = outdir / "shelf_canonical.csv"
    canon_df.to_csv(out_canon, index=False)
    LOG.info(f"Saved canonical mapping: {out_canon}")
    
    # Save non-content shelves
    if noncontent_shelves:
        nc_df = pd.DataFrame(noncontent_shelves)
        out_noncontent = outdir / "noncontent_shelves.csv"
        nc_df.to_csv(out_noncontent, index=False)
        LOG.info(f"Saved non-content shelves: {out_noncontent}")
    
    # Alias detection
    LOG.info("Starting alias detection...")
    alias_candidates = []
    
    # Group shelves by first character for blocking
    blocks = defaultdict(list)
    for shelf in canon_map.values():
        if shelf:
            first_char = shelf[0].lower()
            blocks[first_char].append(shelf)
    
    for first_char, shelves in blocks.items():
        if len(shelves) < 2:
            continue
        
        # Compare all pairs in block
        for i in range(len(shelves)):
            for j in range(i + 1, len(shelves)):
                a, b = shelves[i], shelves[j]
                
                # Quick length filter
                if abs(len(a) - len(b)) > 3:
                    continue
                
                # Calculate similarities
                jw = jaro_winkler_similarity(a, b)
                if jw < (jw_thr - 0.05):  # Early pruning
                    continue
                
                j3 = jaccard_char_ngrams(a, b, 3)
                ed = damerau_levenshtein_distance(a, b)
                
                # Decision logic
                if (jw >= jw_thr and ed <= ed_max) or (jw >= jw_thr and j3 >= j3_thr):
                    decision = "suggest"
                else:
                    decision = "inspect"
                
                alias_candidates.append({
                    'shelf_a': a,
                    'shelf_b': b,
                    'jw': round(jw, 4),
                    'edit': ed,
                    'jaccard3': round(j3, 4),
                    'decision_hint': decision
                })
    
    # Save alias candidates
    if alias_candidates:
        alias_df = pd.DataFrame(alias_candidates)
        alias_df = alias_df.sort_values(['decision_hint', 'jw'], ascending=[True, False])
        out_alias = outdir / "shelf_alias_candidates.csv"
        alias_df.to_csv(out_alias, index=False)
        LOG.info(f"Saved alias candidates: {out_alias}")
    
    # Segmentation
    LOG.info("Starting segmentation...")
    segments_data = []
    segments_vocab = set()
    
    # Build domain lexicon from existing shelves
    domain_lex = set()
    for shelf in canon_map.values():
        if shelf:
            domain_lex.update(shelf.split())
    
    for shelf in canon_map.values():
        if not shelf or len(shelf) < 4:
            continue
        
        # Check for CamelCase
        camel_parts = camel_split(shelf)
        if len(camel_parts) > 1:
            segments = camel_parts
            accepted = True
        else:
            # Check for concatenation
            if len(shelf) >= 6 and shelf.islower():
                # Simple heuristic: try to split at common boundaries
                segments = [shelf]  # Keep as single segment for now
                accepted = False
            else:
                segments = [shelf]
                accepted = True
        
        # Apply guards
        if len(segments) > 1:
            # Guard 1: At least one segment should appear as standalone shelf
            standalone_evidence = sum(1 for seg in segments if seg in domain_lex)
            guard1 = standalone_evidence >= 1
            
            # Guard 2: Lexicon validation
            guard2 = lexicon_ok(segments, zipf_min, domain_lex)
            
            accepted = guard1 and guard2
        
        segments_data.append({
            'shelf_canon': shelf,
            'segments': json.dumps(segments),
            'accepted': accepted,
            'guard1_standalone': standalone_evidence if len(segments) > 1 else 0,
            'guard2_lexicon': guard2 if len(segments) > 1 else True,
            'evidence_count': standalone_evidence if len(segments) > 1 else 0
        })
        
        if accepted:
            segments_vocab.update(segments)
    
    # Save segments
    seg_df = pd.DataFrame(segments_data)
    out_segs = outdir / "shelf_segments.csv"
    seg_df.to_csv(out_segs, index=False)
    LOG.info(f"Saved segments: {out_segs}")
    
    # Save vocabulary
    vocab_sorted = sorted(segments_vocab)
    out_vocab = outdir / "segments_vocab.txt"
    with open(out_vocab, 'w', encoding='utf-8') as f:
        for token in vocab_sorted:
            f.write(f"{token}\n")
    LOG.info(f"Saved vocabulary: {out_vocab}")
    
    # Logging
    logf = open(outdir / "shelf_normalization_log.jsonl", "w", encoding="utf-8")
    
    log_entry = {
        "timestamp": ctx.timestamp,
        "action": "pipeline_complete",
        "git_hash": ctx.git_hash,
        "input_parquet": str(ctx.parsed_parquet),
        "n_unique_shelves": len(shelf_counts),
        "n_canonical": len(canon_map),
        "n_noncontent": len(noncontent_shelves),
        "n_alias_candidates": len(alias_candidates),
        "n_segments": len(segments_data),
        "vocab_size": len(segments_vocab),
        "parameters": {
            "jw_threshold": jw_thr,
            "j3_threshold": j3_thr,
            "edit_max": ed_max,
            "min_seg_evidence": min_seg_evidence,
            "zipf_min": zipf_min,
            "top_k": top_k
        }
    }
    
    logf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    logf.close()

    LOG.info("Wrote: %s, %s, %s, %s, %s",
             out_canon, out_alias, out_segs, out_vocab, out_noncontent)

def main():
    p = argparse.ArgumentParser(description="Shelf normalization (Step 2)")
    p.add_argument("--from-audit", type=Path, required=True,
                   help="Directory containing audit_results.json and overdispersion_tests.json")
    p.add_argument("--parsed-parquet", type=Path, default=None,
                   help="Parsed parquet from Step 1 (overrides audit hint)")
    p.add_argument("--outdir", type=Path, default=Path("shelf_norm_outputs"))
    p.add_argument("--n-top-shelves", type=int, default=100000,
                   help="Limit to top-K shelves by frequency for aliasing/segmentation (default: 100k)")
    p.add_argument("--jw-threshold", type=float, default=0.94)
    p.add_argument("--j3-threshold", type=float, default=0.80)
    p.add_argument("--edit-max", type=int, default=1)
    p.add_argument("--min-seg-evidence", type=int, default=5)
    p.add_argument("--zipf-min", type=float, default=3.0)
    p.add_argument("--verbose-logs", action="store_true")
    args = p.parse_args()

    set_logging(quiet=not args.verbose_logs)
    ctx = load_audit_context(args.from_audit, args.parsed_parquet)
    run_pipeline(
        ctx=ctx,
        outdir=args.outdir,
        jw_thr=args.jw_threshold,
        j3_thr=args.j3_threshold,
        ed_max=args.edit_max,
        min_seg_evidence=args.min_seg_evidence,
        zipf_min=args.zipf_min,
        top_k=args.n_top_shelves,
        verbose_logs=args.verbose_logs,
    )

if __name__ == "__main__":
    main()
