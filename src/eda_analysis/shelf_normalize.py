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
project_root = Path(__file__).parent.parent.parent
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
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Silence matplotlib etc. unless explicitly asked
    try:
        import matplotlib
        matplotlib.set_loglevel("warning")
    except Exception:
        pass

# ---------- audit context ----------
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
    audit_json = audit_dir / "audit_results.json"
    overdisp = audit_dir / "overdispersion_tests.json"

    if not audit_json.exists():
        raise FileNotFoundError(f"Missing {audit_json}")
    if not overdisp.exists():
        LOG.warning("Missing overdispersion_tests.json (provenance only).")

    git_hash = None
    ts = None
    try:
        j = json.loads(audit_json.read_text())
        git_hash = j.get("repo_state", {}).get("git_hash")
        ts = j.get("timestamp")
        if parsed_parquet is None:
            # try to find recommended parsed path in audit json
            parsed_parquet = j.get("artifacts", {}).get("parsed_parquet")
            parsed_parquet = Path(parsed_parquet) if parsed_parquet else None
    except Exception as e:
        LOG.warning("Could not parse audit_results.json: %s", e)

    if parsed_parquet is None:
        raise ValueError("You must provide --parsed-parquet or have it in audit_results.json")

    return AuditContext(
        audit_dir=audit_dir,
        audit_json=audit_json,
        overdisp_json=overdisp,
        parsed_parquet=Path(parsed_parquet),
        git_hash=git_hash,
        timestamp=ts,
    )

# ---------- leakage filters ----------
def is_noncontent(shelf: str) -> Optional[str]:
    """Return category name if matches a leakage pattern; else None."""
    s = shelf.strip().lower()
    
    # Reading status patterns
    reading_status = {
        'to-read', 'to read', 'tbr', 'want-to-read', 'want to read', 'wtr',
        'currently-reading', 'currently reading', 'reading', 'cr',
        'read', 'finished', 'done', 'completed',
        'dnf', 'did-not-finish', 'did not finish', 'abandoned',
        'on-hold', 'on hold', 'paused', 'suspended'
    }
    
    # Ownership/collection patterns  
    ownership = {
        'owned', 'have', 'have-it', 'have it', 'bought', 'purchased',
        'library', 'borrowed', 'borrow', 'lent', 'loan',
        'kindle', 'ebook', 'e-book', 'audiobook', 'audio-book',
        'hardcover', 'paperback', 'mass-market', 'mass market',
        'wishlist', 'wish-list', 'want', 'want-it', 'want it'
    }
    
    # Date/reading challenge patterns
    date_patterns = {
        '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025',
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december',
        'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
        'summer', 'winter', 'spring', 'fall', 'autumn'
    }
    
    # Pure punctuation/dash runs
    if re.match(r'^[-_\.\s]+$', s) or len(s) <= 2:
        return 'punctuation'
    
    # Check exact matches
    if s in reading_status:
        return 'reading_status'
    if s in ownership:
        return 'ownership'
    if s in date_patterns:
        return 'date_pattern'
    
    # Check patterns with dashes/underscores normalized
    s_norm = re.sub(r'[-_\s]+', '-', s)
    if s_norm in reading_status or s_norm in ownership:
        return 'reading_status' if s_norm in reading_status else 'ownership'
    
    # Check if starts with common prefixes
    if any(s.startswith(prefix) for prefix in ['to-', 'want-', 'have-', 'currently-']):
        return 'reading_status'
    
    return None

# ---------- canonicalization (2.1.A) ----------
_WS_RE = re.compile(r"\s+")
_SEP_STD = re.compile(r"(?<=\w)[-_]+(?=\w)")  # replace with space when alnum on both sides
_EDGE_PUNCT = re.compile(r"^[\W_]+|[\W_]+$")

def canon_key(raw: str) -> str:
    # normalize → casefold → collapse whitespace
    txt = unicodedata.normalize("NFKC", raw)
    txt = txt.strip()
    # keep original for display, but build comparison key:
    key = _SEP_STD.sub(" ", txt)  # hyphen/underscore standardization (guarded)
    key = _EDGE_PUNCT.sub("", key)  # drop edge punctuation
    key = _WS_RE.sub(" ", key).strip()
    # for comparison, strip diacritics
    key = unicodedata.normalize("NFD", key)
    key = "".join(ch for ch in key if not unicodedata.combining(ch))
    key = key.casefold()
    return key

# ---------- segmentation (2.2) ----------
CAMEL_HIT = re.compile(r"[A-Z][a-z]+[A-Z]")  # looks like CamelCase somewhere
LOWER_CONCAT = re.compile(r"^[a-z]{6,}$")    # long lowercase run (tune)

def camel_split(token: str) -> List[str]:
    # Split on boundaries between lower->Upper or digit->Upper; keep order
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", token)
    return parts if parts else [token]

def zipf_ok(w: str, zipf_min: float) -> bool:
    if not WORDFREQ_AVAILABLE:
        # Fallback: simple heuristic based on length and common patterns
        return len(w) >= 3 and not w.isdigit()
    # allow domain short tokens explicitly later (e.g., mc, ya)
    return zipf_frequency(w, "en") >= zipf_min

def lexicon_ok(parts: List[str], zipf_min: float, domain_lex: Set[str]) -> bool:
    for p in parts:
        p_l = p.lower()
        if p_l in domain_lex:
            continue
        if not zipf_ok(p_l, zipf_min):
            return False
    return True

# ---------- alias metrics (2.1.B) ----------
def jaccard_char_ngrams(a: str, b: str, n: int = 3) -> float:
    def ngrams(s: str) -> Set[str]:
        s = f" {s} "  # padding helps start/end stability
        return {s[i:i+n] for i in range(len(s)-n+1)}
    A, B = ngrams(a), ngrams(b)
    if not A and not B:
        return 1.0
    return len(A & B) / max(1, len(A | B))

def simple_edit_distance(a: str, b: str) -> int:
    """Simple Levenshtein distance fallback if rapidfuzz not available."""
    if len(a) < len(b):
        a, b = b, a
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
    """Jaro-Winkler similarity fallback if rapidfuzz not available."""
    if not RAPIDFUZZ_AVAILABLE:
        # Simple fallback based on character overlap
        if a == b:
            return 1.0
        if len(a) == 0 or len(b) == 0:
            return 0.0
        
        # Simple character-based similarity
        a_chars = set(a.lower())
        b_chars = set(b.lower())
        intersection = len(a_chars & b_chars)
        union = len(a_chars | b_chars)
        return intersection / union if union > 0 else 0.0
    
    return JaroWinkler.similarity(a, b) / 100.0

def damerau_levenshtein_distance(a: str, b: str) -> int:
    """Damerau-Levenshtein distance with fallback."""
    if not RAPIDFUZZ_AVAILABLE:
        return simple_edit_distance(a, b)
    return DamerauLevenshtein.distance(a, b)

# ---------- I/O helpers ----------
def stream_unique_shelves(parquet_path: Path,
                          col: str = "shelves",
                          nrows: Optional[int] = None) -> Counter:
    """Count unique shelf strings from list column."""
    counts = Counter()
    # Using pyarrow for speed + streaming
    tbl = pq.read_table(parquet_path)
    df = tbl.to_pandas(types_mapper=pd.ArrowDtype)  # preserve Arrow types
    series = df[col]
    for arr in series:
        # arr might be list, numpy array, or pyarrow ListArray element
        if arr is None:
            continue
        if hasattr(arr, "to_pylist"):
            items = arr.to_pylist()
        elif isinstance(arr, (list, tuple, np.ndarray)):
            items = list(arr)
        else:
            # last resort
            items = list(arr)
        for s in items:
            if isinstance(s, str) and s.strip():
                counts[s] += 1
    if nrows:
        # (kept for API compatibility; we read whole file above)
        pass
    return counts

# ---------- main pipeline ----------
def run_pipeline(ctx: AuditContext,
                 outdir: Path,
                 jw_thr: float,
                 j3_thr: float,
                 ed_max: int,
                 min_seg_evidence: int,
                 zipf_min: float,
                 top_k: Optional[int],
                 verbose_logs: bool):

    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "shelf_normalization_log.jsonl"
    logf = log_path.open("w", encoding="utf-8")

    # 1) No filters - process all shelves as content

    # 2) Load shelves universe
    counts = stream_unique_shelves(ctx.parsed_parquet)
    total_unique = len(counts)
    LOG.info("Loaded %d unique shelves (total tokens=%d)",
             total_unique, sum(counts.values()))

    # (optional) focus on top-K frequent shelves for alias discovery
    shelves_sorted = [s for s, _ in counts.most_common(top_k or total_unique)]

    # 3) Canonicalization
    canon_map = {}            # raw -> canon
    canon_reason = {}         # raw -> reason
    noncontent_flag = {}      # raw -> category or None
    canon_counts = Counter()  # canon -> freq

    # Track canonicalization statistics
    canon_stats = {
        "as_is": 0,
        "casefold": 0,
        "sep_normalize": 0,
        "punct_trim": 0,
        "noncontent_filtered": 0
    }
    
    for raw in shelves_sorted:
        leak = is_noncontent(raw)
        key = canon_key(raw)
        reason = []
        
        # Track specific transformations
        if key != raw:
            if raw.lower() != key:
                reason.append("casefold")
                canon_stats["casefold"] += 1
            if re.sub(r'[-_]+', ' ', raw) != raw:
                reason.append("sep_normalize")
                canon_stats["sep_normalize"] += 1
            if re.sub(r'^[\W_]+|[\W_]+$', '', raw) != raw:
                reason.append("punct_trim")
                canon_stats["punct_trim"] += 1
        else:
            canon_stats["as_is"] += 1
            
        if leak:
            noncontent_flag[raw] = leak
            canon_stats["noncontent_filtered"] += 1
            
        canon_map[raw] = key
        canon_reason[raw] = ";".join(reason) if reason else "as_is"
        canon_counts[key] += counts[raw]

        # provenance log
        rec = {
            "stage": "canonicalize",
            "raw": raw,
            "canon": key,
            "noncontent_category": leak,
            "count": counts[raw],
            "transformation": ";".join(reason) if reason else "as_is"
        }
        logf.write(json.dumps(rec) + "\n")

    # 4) Segmentation (DFKI conservative)
    # Build corpus evidence: which tokens occur as standalone shelves?
    standalone = set(canon_counts.keys())

    # Domain lexicon: frequent shelf tokens & description unigrams/bigrams could be added here.
    # For now, bootstrap from shelf tokens themselves (split on spaces/hyphens).
    domain_tokens = Counter()
    for c in standalone:
        for t in re.split(r"[ \-]", c):
            if t:
                domain_tokens[t] += 1
    domain_lex = {t for t, f in domain_tokens.items() if f >= min_seg_evidence}

    seg_records = []  # for CSV
    seg_vocab = set()
    
    # Track segmentation statistics
    seg_stats = {
        "total_processed": 0,
        "has_whitespace": 0,
        "too_short": 0,
        "common_word": 0,
        "not_camel_or_concat": 0,
        "camel_case": 0,
        "lowercase_concat": 0,
        "guard1_failed": 0,
        "guard2_failed": 0,
        "zipf_failed": 0,
        "accepted": 0
    }

    def maybe_segment(canon: str) -> Tuple[List[str], bool, bool, bool]:
        """Return (segments, accepted, guard1, guard2)"""
        original = canon
        seg_stats["total_processed"] += 1
        
        # Gate 1: Must have no whitespace
        if " " in original:
            seg_stats["has_whitespace"] += 1
            return [original], False, False, False
            
        # Gate 2: Must be length >= 6
        if len(original) < 6:
            seg_stats["too_short"] += 1
            return [original], False, False, False
            
        # Gate 3: Check if already a common word (Zipf >= 3.0)
        if WORDFREQ_AVAILABLE and zipf_frequency(original.lower(), "en") >= zipf_min:
            seg_stats["common_word"] += 1
            return [original], False, False, False
            
        # Gate 4: Must be CamelCase OR all-lowercase concatenation
        looks_camel = bool(CAMEL_HIT.search(original))
        looks_concat = bool(LOWER_CONCAT.fullmatch(original))
        
        if not (looks_camel or looks_concat):
            seg_stats["not_camel_or_concat"] += 1
            return [original], False, False, False

        pieces = [original]
        if looks_camel:
            pieces = camel_split(original)
            seg_stats["camel_case"] += 1
        elif looks_concat:
            seg_stats["lowercase_concat"] += 1
            # DP over lowercased string with wordfreq costs
            s = original.lower()
            n = len(s)
            best = [math.inf]*(n+1)
            back = [-1]*(n+1)
            best[0] = 0.0
            for i in range(n):
                if best[i] == math.inf:
                    continue
                for j in range(i+2, min(n, i+15)+1):  # word len 2..15
                    w = s[i:j]
                    if WORDFREQ_AVAILABLE:
                        cost = -zipf_frequency(w, "en")  # lower cost for frequent words
                    else:
                        # Simple fallback cost based on length
                        cost = -len(w) if len(w) >= 3 else math.inf
                    if cost == math.inf:
                        continue
                    if best[i] + cost < best[j]:
                        best[j] = best[i] + cost
                        back[j] = i
            if back[n] != -1:
                parts = []
                k = n
                while k > 0:
                    i = back[k]
                    parts.append(s[i:k])
                    k = i
                pieces = list(reversed(parts))

        # Guards (DFKI): Guard1 = any piece occurs as standalone shelf; Guard2 = lexicon membership
        guard1 = any(p.lower() in standalone for p in pieces)
        guard2 = lexicon_ok([p.lower() for p in pieces], zipf_min=zipf_min, domain_lex=domain_lex)

        # Accept rule: keep split iff it yields >=2 tokens AND each has Zipf >= 3.0
        accepted = guard1 and guard2 and len(pieces) > 1
        if accepted and WORDFREQ_AVAILABLE:
            # Double-check each piece meets Zipf threshold
            for piece in pieces:
                if zipf_frequency(piece.lower(), "en") < zipf_min:
                    accepted = False
                    seg_stats["zipf_failed"] += 1
                    break
        
        # Track guard failures
        if not guard1:
            seg_stats["guard1_failed"] += 1
        if not guard2:
            seg_stats["guard2_failed"] += 1
        if accepted:
            seg_stats["accepted"] += 1
                    
        segs = [p.lower() for p in pieces] if accepted else [original]

        return segs, accepted, guard1, guard2

    for c, freq in canon_counts.items():
        segs, accepted, g1, g2 = maybe_segment(c)
        if accepted:
            for s in segs:
                seg_vocab.add(s)
        seg_records.append({
            "shelf_canon": c,
            "segments": " ".join(segs),
            "accepted": bool(accepted),
            "guard1_standalone": bool(g1),
            "guard2_lexicon": bool(g2),
            "evidence_count": int(freq),
        })
        logf.write(json.dumps({
            "stage": "segment",
            "canon": c,
            "segments": segs,
            "accepted": accepted,
            "guard1": g1, "guard2": g2,
        }) + "\n")

    # 5) Alias candidate suggestions (2.1.B), run on segmented forms (post-split)
    # Robust blocking: Q-gram index + sorted-neighborhood windowing
    
    def qgram_signature(s: str, q: int = 3) -> Set[str]:
        """Generate Q-gram signature for blocking."""
        s = s.lower().strip()
        if len(s) < q:
            return {s}
        return {s[i:i+q] for i in range(len(s) - q + 1)}
    
    def sorted_neighborhood_pairs(items: List[str], window: int = 50) -> Iterable[Tuple[str, str]]:
        """Generate pairs within sorted neighborhood window."""
        sorted_items = sorted(items)
        for i in range(len(sorted_items)):
            for j in range(i+1, min(i+window+1, len(sorted_items))):
                yield sorted_items[i], sorted_items[j]
    
    def canopy_clustering(items: List[str], loose_thresh: float = 0.8, tight_thresh: float = 0.9) -> List[List[str]]:
        """Simple canopy clustering for blocking."""
        canopies = []
        remaining = set(items)
        
        while remaining:
            # Pick a random item as canopy center
            center = remaining.pop()
            canopy = [center]
            
            # Add items within loose threshold
            to_remove = set()
            for item in remaining:
                j3 = jaccard_char_ngrams(center, item, n=3)
                if j3 >= loose_thresh:
                    canopy.append(item)
                    to_remove.add(item)
            
            remaining -= to_remove
            canopies.append(canopy)
        
        return canopies

    canon_list = [r["segments"] if r["accepted"] else r["shelf_canon"] for r in seg_records]
    canon_list = list(dict.fromkeys(canon_list))  # preserve order, uniq
    
    LOG.info("Building alias candidates from %d canonical shelves...", len(canon_list))
    
    # Method 1: Q-gram blocking
    qgram_buckets = defaultdict(list)
    for s in canon_list:
        sig = qgram_signature(s, q=3)
        # Use first few q-grams as bucket key
        key = tuple(sorted(list(sig))[:5])  # Use first 5 q-grams
        qgram_buckets[key].append(s)
    
    # Method 2: Sorted neighborhood (for similar-length items)
    length_buckets = defaultdict(list)
    for s in canon_list:
        length_buckets[len(s)].append(s)
    
    # Method 3: Canopy clustering (for high-similarity groups)
    canopies = canopy_clustering(canon_list, loose_thresh=0.7, tight_thresh=0.85)
    
    alias_rows = []
    seen_pairs = set()
    
    # Process Q-gram buckets
    for bucket in qgram_buckets.values():
        if len(bucket) < 2:
            continue
        for a, b in sorted_neighborhood_pairs(bucket, window=20):
            if (a, b) in seen_pairs or (b, a) in seen_pairs:
                continue
            seen_pairs.add((a, b))
            
            # Quick length gate
            if abs(len(a) - len(b)) > 3:
                continue
                
            jw = jaro_winkler_similarity(a, b)
            ed = damerau_levenshtein_distance(a, b)
            j3 = jaccard_char_ngrams(a, b, n=3)
            
            # Tiered thresholds
            if len(a) <= 6 and len(b) <= 6:
                # Short strings: Damerau-Levenshtein <= 1
                if ed <= 1:
                    decision = "suggest"
                else:
                    continue
            elif len(a) <= 12 and len(b) <= 12:
                # Mid-length: Jaro-Winkler >= 0.92
                if jw >= 0.92:
                    decision = "suggest"
                else:
                    continue
            else:
                # Long strings: token_set_ratio >= 95 (approximated with j3)
                if j3 >= 0.95:
                    decision = "suggest"
                else:
                    continue
            
            alias_rows.append({
                "shelf_a": a, "shelf_b": b,
                "jw": round(jw, 4), "edit": int(ed), "jaccard3": round(j3, 4),
                "decision_hint": decision
            })
            logf.write(json.dumps({
                "stage": "alias",
                "a": a, "b": b, "jw": jw, "ed": int(ed), "j3": j3,
                "decision": decision
            }) + "\n")
    
    # Process length buckets (for similar-length items)
    for length, bucket in length_buckets.items():
        if len(bucket) < 2:
            continue
        for a, b in sorted_neighborhood_pairs(bucket, window=30):
            if (a, b) in seen_pairs or (b, a) in seen_pairs:
                continue
            seen_pairs.add((a, b))
            
            jw = jaro_winkler_similarity(a, b)
            ed = damerau_levenshtein_distance(a, b)
            j3 = jaccard_char_ngrams(a, b, n=3)
            
            # Apply same tiered thresholds
            if len(a) <= 6 and len(b) <= 6:
                if ed <= 1:
                    decision = "suggest"
                else:
                    continue
            elif len(a) <= 12 and len(b) <= 12:
                if jw >= 0.92:
                    decision = "suggest"
                else:
                    continue
            else:
                if j3 >= 0.95:
                    decision = "suggest"
                else:
                    continue
            
            alias_rows.append({
                "shelf_a": a, "shelf_b": b,
                "jw": round(jw, 4), "edit": int(ed), "jaccard3": round(j3, 4),
                "decision_hint": decision
            })
            logf.write(json.dumps({
                "stage": "alias",
                "a": a, "b": b, "jw": jw, "ed": int(ed), "j3": j3,
                "decision": decision
            }) + "\n")
    
    LOG.info("Generated %d alias candidate pairs", len(alias_rows))

    # 6) Write artifacts (2.3)
    out_canon = outdir / "shelf_canonical.csv"
    pd.DataFrame({
        "shelf_raw": list(canon_map.keys()),
        "shelf_canon": [canon_map[k] for k in canon_map.keys()],
        "reason": [canon_reason[k] for k in canon_map.keys()],
        "noncontent_category": [noncontent_flag.get(k) for k in canon_map.keys()],
        "count": [counts[k] for k in canon_map.keys()],
    }).to_csv(out_canon, index=False)

    out_alias = outdir / "shelf_alias_candidates.csv"
    if alias_rows:
        pd.DataFrame(alias_rows).sort_values(["decision_hint","jw","jaccard3"], ascending=[True, False, False]
        ).to_csv(out_alias, index=False)
    else:
        # Create empty DataFrame with correct columns
        pd.DataFrame(columns=["shelf_a", "shelf_b", "jw", "edit", "jaccard3", "decision_hint"]).to_csv(out_alias, index=False)

    out_segs = outdir / "shelf_segments.csv"
    pd.DataFrame(seg_records).to_csv(out_segs, index=False)

    out_vocab = outdir / "segments_vocab.txt"
    out_vocab.write_text("\n".join(sorted(seg_vocab)))

    # noncontent list (for downstream exclusion)
    out_noncontent = outdir / "noncontent_shelves.csv"
    nonc_rows = [{"shelf_raw": k, "category": v, "count": counts[k]}
                 for k, v in noncontent_flag.items()]
    if nonc_rows:
        pd.DataFrame(nonc_rows).to_csv(out_noncontent, index=False)
    else:
        # Create empty DataFrame with correct columns
        pd.DataFrame(columns=["shelf_raw", "category", "count"]).to_csv(out_noncontent, index=False)

    # 7) Close log with provenance footer
    footer = {
        "stage": "provenance",
        "audit_dir": str(ctx.audit_dir),
        "audit_results_json": str(ctx.audit_json),
        "overdispersion_json": str(ctx.overdisp_json),
        "parsed_parquet": str(ctx.parsed_parquet),
        "git_hash": ctx.git_hash,
        "audit_timestamp": ctx.timestamp,
        "run_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "thresholds": {
            "jw": jw_thr, "j3": j3_thr, "edit_max": ed_max,
            "min_seg_evidence": min_seg_evidence, "zipf_min": zipf_min
        },
        "dependencies": {
            "rapidfuzz_available": RAPIDFUZZ_AVAILABLE,
            "wordfreq_available": WORDFREQ_AVAILABLE
        },
        "statistics": {
            "canonicalization": canon_stats,
            "segmentation": seg_stats,
            "alias_candidates": len(alias_rows),
            "total_shelves_processed": len(shelves_sorted),
            "unique_canonical": len(canon_counts),
            "compression_ratio": len(canon_counts) / len(shelves_sorted) if shelves_sorted else 0.0
        }
    }
    logf.write(json.dumps(footer) + "\n")
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
