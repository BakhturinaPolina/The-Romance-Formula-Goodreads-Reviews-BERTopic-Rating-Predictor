"""
Romance Book Data Extraction Tool

A CLI tool that loads the AlekseyKorshuk/romance-books dataset from Hugging Face
and extracts (author, title) pairs using layered heuristics with confidence scoring.

Usage:
    python -m src.external_data_extraction.extract_romance_books --out output.csv --limit 1000
    python -m src.external_data_extraction.extract_romance_books --out output.jsonl --use-spacy
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable, List, Optional, Tuple, Dict

try:
    from datasets import load_dataset
except ImportError as e:
    print("Install required dependency: pip install datasets", file=sys.stderr)
    raise

# --------- Regular Expressions ---------
TITLE_SEP_RE = re.compile(r"\s+(?:by)\s+", re.IGNORECASE)
DASH_SPLIT_RE = re.compile(r"\s+[-–—]\s+")
LEADING_QUOTES_RE = re.compile(r"""^[\s"'""'«»\[\(]+|[\s"'""'«»\]\)]+$""")
MULTISPACE_RE = re.compile(r"\s+")
QUOTED_TITLE_RE = re.compile(r"""^[""](?P<title>[^""]+)[""]\s*(?:by\s+(?P<author>[^|\-\(\[]+))?""", re.IGNORECASE)
PARENS_RE = re.compile(r"\s*[\(\[].*?[\)\]]\s*")
TRAILING_URL_ARTIFACTS_RE = re.compile(r"[_\-]+(pdf|epub|mobi|txt)$", re.IGNORECASE)
URL_TOKEN_SPLIT_RE = re.compile(r"[_\-\+]+")
NON_WORD_RE = re.compile(r"[^\w\s\.\-']+")

NAME_TOKEN_RE = re.compile(r"^[A-Z][a-z]+\.?$|^[A-Z]\.$|^O'[A-Z][a-z]+$|^Mc[A-Z][a-z]+$")
LOWER_WORD_RE = re.compile(r".*[a-z].*")


@dataclass
class Extraction:
    """Result of author/title extraction with metadata."""
    title: Optional[str]
    author: Optional[str]
    method: str
    confidence: float
    debug: Dict[str, str]


def norm_space(s: str) -> str:
    """Normalize whitespace in string."""
    return MULTISPACE_RE.sub(" ", s).strip()


def clean_field(s: str) -> str:
    """Clean and normalize a field value."""
    if not s:
        return ""
    
    s = s.strip()
    s = PARENS_RE.sub(" ", s)
    s = LEADING_QUOTES_RE.sub("", s)
    s = LEADING_QUOTES_RE.sub("", s)  # two passes for nested quotes
    s = NON_WORD_RE.sub(lambda m: " " if m.group(0).strip() else "", s)
    s = norm_space(s)
    
    # Heuristic: Title-case excessive ALL-CAPS words (rare; keeps acronyms)
    if len(s) > 0 and sum(ch.isupper() for ch in s) / max(1, sum(ch.isalpha() for ch in s)) > 0.75:
        s = " ".join(w.capitalize() if len(w) > 2 else w for w in s.split())
    
    return s


def is_name_like(segment: str) -> bool:
    """Check if a segment looks like a person's name."""
    seg = segment.strip().strip("-–—")
    if not seg or len(seg.split()) > 5 or len(seg.split()) < 2:
        return False
    
    tokens = seg.replace(",", " ").split()
    good = 0
    
    for t in tokens:
        if NAME_TOKEN_RE.match(t):
            good += 1
        elif t.istitle() and LOWER_WORD_RE.match(t):
            good += 1
        elif t.lower() in {"de", "da", "del", "van", "der", "von", "le", "la"}:
            good += 1
        elif t in {"&", "and"}:  # co-authors
            good += 1
    
    return good >= max(2, math.ceil(0.6 * len(tokens)))


def guess_title_from_url(url: str) -> Optional[str]:
    """Extract a potential title from URL path."""
    if not url:
        return None
    
    try:
        path = PurePosixPath(url.split("://", 1)[-1])
    except Exception:
        return None
    
    stem = path.name or ""
    if "." in stem:
        stem = stem.rsplit(".", 1)[0]
    
    stem = TRAILING_URL_ARTIFACTS_RE.sub("", stem)
    parts = [p for p in URL_TOKEN_SPLIT_RE.split(stem) if p and not p.isdigit()]
    
    # Filter junky hosts like bookrix, wattpad, etc., keep meaningful words
    bad = {"bookrix", "wattpad", "chapter", "ch", "part", "novel", "romance", "books", "story", "fanfic"}
    parts = [p for p in parts if p.lower() not in bad]
    
    if len(parts) >= 2:
        title = " ".join(parts).strip()
        return clean_field(title)
    
    return None


# --------- Extraction Methods (ordered by preference) ---------

def extract_by_clause(header: str) -> Optional[Extraction]:
    """Extract 'Title by Author' pattern."""
    if not TITLE_SEP_RE.search(header):
        return None
    
    left, right = TITLE_SEP_RE.split(header, maxsplit=1)
    title = clean_field(left)
    author = clean_field(right.split("|")[0].split("-")[0])
    
    if title and author and is_name_like(author):
        return Extraction(title, author, "title_by_author", 0.92, {"left": left, "right": right})
    
    return None


def extract_dash_split(header: str) -> Optional[Extraction]:
    """Extract 'Author — Title' or 'Title — Author' pattern."""
    if not DASH_SPLIT_RE.search(header):
        return None
    
    parts = DASH_SPLIT_RE.split(header, maxsplit=1)
    if len(parts) != 2:
        return None
    
    a, b = (clean_field(parts[0]), clean_field(parts[1]))
    
    # Decide which side is name-like
    if is_name_like(a) and not is_name_like(b):
        return Extraction(b, a, "dash_author_right", 0.85, {})
    if is_name_like(b) and not is_name_like(a):
        return Extraction(a, b, "dash_author_left", 0.85, {})
    
    return None


def extract_quoted(header: str) -> Optional[Extraction]:
    """Extract quoted title with optional 'by Author'."""
    m = QUOTED_TITLE_RE.match(header.strip())
    if not m:
        return None
    
    title = clean_field(m.group("title") or "")
    author = clean_field(m.group("author") or "")
    
    if title and author and is_name_like(author):
        return Extraction(title, author, "quoted_title_by", 0.88, {})
    if title and not author:
        return Extraction(title, None, "quoted_title_only", 0.6, {})
    
    return None


def extract_leading_name(header: str) -> Optional[Extraction]:
    """Extract 'Author Title' pattern (name at start)."""
    # e.g., "Cherie Benjamin A Sinless Betrayal (completed)"
    tokens = header.strip().split()
    if len(tokens) < 3:
        return None
    
    # try 2..4 token name at start
    for k in (4, 3, 2):
        if len(tokens) <= k:
            continue
        name = " ".join(tokens[:k])
        rest = " ".join(tokens[k:])
        if is_name_like(name) and len(rest.split()) >= 2:
            title = clean_field(rest)
            author = clean_field(name)
            return Extraction(title, author, "leading_name_then_title", 0.78, {"k": str(k)})
    
    return None


def finalize_with_url_fallback(ex: Optional[Extraction], url: str, header: str) -> Optional[Extraction]:
    """Apply URL fallback and confidence adjustments."""
    # If no extraction, or missing title/author, try url title; bump confidence if consistent
    url_title = guess_title_from_url(url) or ""
    
    if ex is None:
        if url_title:
            return Extraction(url_title, None, "url_title_only", 0.55, {})
        # As absolute last resort, take first 6 words as title-ish
        rough = clean_field(" ".join(header.split()[:6]))
        if rough:
            return Extraction(rough, None, "header_snippet", 0.4, {})
        return None
    
    # fill missing title
    if (not ex.title) and url_title:
        ex.title = url_title
        ex.confidence = max(ex.confidence, 0.65)
        ex.method += "+url_title"
    
    # small corroboration bump if header starts with title
    if ex.title and header.lower().startswith(ex.title.lower()[:max(5, min(20, len(ex.title)))]):
        ex.confidence = min(0.99, ex.confidence + 0.03)
    
    return ex


def enrich_with_spacy(ex: Optional[Extraction], header: str, use_spacy: bool) -> Optional[Extraction]:
    """Optional spaCy PERSON NER enrichment."""
    if not use_spacy:
        return ex
    
    try:
        import spacy  # heavy; optional
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        return ex
    
    doc = nlp(header)
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    
    if not persons:
        return ex
    
    # Prefer first reasonably long person name
    candidate = None
    for p in persons:
        if len(p.split()) >= 2 and len(p) >= 5:
            candidate = p
            break
    candidate = candidate or persons[0]
    
    if ex is None:
        return Extraction(None, clean_field(candidate), "spacy_person_only", 0.6, {})
    
    if not ex.author and candidate:
        ex.author = clean_field(candidate)
        ex.method += "+spacy_author"
        ex.confidence = max(ex.confidence, 0.7)
    
    return ex


def pick_text_column(example: dict) -> Optional[str]:
    """Find the most likely text column in the dataset."""
    # Try common text-ish fields
    priorities = ["text", "content", "body", "raw", "document", "data", "excerpt"]
    for key in priorities:
        if key in example and isinstance(example[key], str) and example[key].strip():
            return key
    
    # fallback: longest string field
    best = None
    for k, v in example.items():
        if isinstance(v, str) and v.strip():
            if best is None or len(v) > len(example[best]):
                best = k
    
    return best


def build_header(text: str) -> str:
    """Extract header text from full text content."""
    # first non-empty line, stripped of surrounding noise
    for line in text.splitlines():
        ln = norm_space(line)
        if ln:
            return ln[:200]
    
    return norm_space(text)[:200]


def extract_one(example: dict, use_spacy: bool = False) -> Extraction:
    """Extract author/title from a single dataset example."""
    url = example.get("url") or example.get("source") or example.get("link") or ""
    text_col = pick_text_column(example) or ""
    raw = example.get(text_col, "") if text_col else ""
    header = build_header(raw)

    # layered extraction
    ex: Optional[Extraction] = None
    for fn in (extract_by_clause, extract_dash_split, extract_quoted, extract_leading_name):
        ex = fn(header)
        if ex:
            break
    
    ex = finalize_with_url_fallback(ex, url, header)
    ex = enrich_with_spacy(ex, header, use_spacy)
    
    if ex is None:
        ex = Extraction(None, None, "failed", 0.0, {})
    
    # clamp + clean
    ex.title = clean_field(ex.title) if ex.title else None
    ex.author = clean_field(ex.author) if ex.author else None
    
    return ex


# --------- I/O Functions ---------

def write_csv(rows: List[dict], path: str) -> None:
    """Write results to CSV file."""
    fieldnames = ["idx", "author", "title", "confidence", "method", "source_url", "raw_header"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def write_jsonl(rows: List[dict], path: str) -> None:
    """Write results to JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def summarize(rows: List[dict]) -> str:
    """Generate summary statistics."""
    n = len(rows)
    ok = sum(1 for r in rows if r.get("author") and r.get("title"))
    semi = sum(1 for r in rows if (r.get("author") or r.get("title")) and not (r.get("author") and r.get("title")))
    hi = sum(1 for r in rows if r.get("confidence", 0) >= 0.85)
    lo = sum(1 for r in rows if r.get("confidence", 0) < 0.6)
    
    return (
        f"rows={n}, full_pairs={ok} ({ok/n:.1%}), partial={semi} ({semi/n:.1%}), "
        f"high_conf={hi} ({hi/n:.1%}), low_conf={lo} ({lo/n:.1%})"
    )


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    ap = argparse.ArgumentParser(description="Extract (author, title) from AlekseyKorshuk/romance-books")
    ap.add_argument("--split", default="train", help="dataset split (default: train)")
    ap.add_argument("--limit", type=int, default=0, help="max rows (0 = all)")
    ap.add_argument("--out", required=True, help="output file path (CSV or JSONL by extension)")
    ap.add_argument("--fmt", choices=["csv", "jsonl"], default=None, help="force output format")
    ap.add_argument("--use-spacy", action="store_true", help="enable spaCy PERSON fallback")
    args = ap.parse_args(argv)

    # Load dataset (user must accept HF gated terms if any)
    print("Loading dataset from Hugging Face...", file=sys.stderr)
    ds = load_dataset("AlekseyKorshuk/romance-books", split=args.split)  # requires `datasets`
    total = len(ds)
    lim = args.limit if args.limit and args.limit > 0 else total

    print(f"Processing {lim} rows from {total} total...", file=sys.stderr)
    
    rows = []
    # Detect text column once using first example
    example0 = ds[0] if total else {}
    text_col = pick_text_column(example0) or "text"
    url_col = "url" if "url" in example0 else ("source" if "source" in example0 else ("link" if "link" in example0 else None))

    for i, ex in enumerate(ds.take(lim) if hasattr(ds, "take") else (ds[j] for j in range(lim))):
        if i % 1000 == 0 and i > 0:
            print(f"Processed {i} rows...", file=sys.stderr)
        
        header = build_header(ex.get(text_col, "") if text_col else "")
        res = extract_one(ex, args.use_spacy)
        row = {
            "idx": i,
            "author": res.author or "",
            "title": res.title or "",
            "confidence": round(res.confidence, 3),
            "method": res.method,
            "source_url": ex.get(url_col, "") if url_col else (ex.get("url") or ex.get("source") or ex.get("link") or ""),
            "raw_header": header,
        }
        rows.append(row)

    # Write
    fmt = args.fmt or (args.out.lower().endswith(".jsonl") and "jsonl") or "csv"
    print(f"Writing {len(rows)} rows to {args.out} ({fmt})...", file=sys.stderr)
    
    if fmt == "csv":
        write_csv(rows, args.out)
    else:
        write_jsonl(rows, args.out)

    # Report
    print(summarize(rows))
    
    # Show a few low-confidence rows for manual QA
    bad = [r for r in rows if r["confidence"] < 0.6][:10]
    if bad:
        print("\nLow-confidence samples (<=10):")
        for r in bad:
            print(f"- idx={r['idx']} conf={r['confidence']} method={r['method']} | header='{r['raw_header']}' | url='{r['source_url']}'")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
