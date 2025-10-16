"""
Fuzzy title-author matching between Goodreads subset and external text datasets.

This module provides functionality to match books from a Goodreads dataset
to external text datasets (like Hugging Face datasets) using fuzzy string
matching on titles and authors with configurable thresholds and blocking
for efficiency.

Author: Research Assistant
Date: 2025-01-09
"""

import re
import sys
import math
import json
import string
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import pandas as pd
from rapidfuzz import fuzz, process
from unidecode import unidecode

# Optional HF dataset (comment out if you already have a local CSV of texts)
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False
    warnings.warn("datasets (HF) not available. You can load your text corpus from CSV instead.")


# ----------------------------
# Config / thresholds
# ----------------------------
@dataclass
class MatchConfig:
    """Configuration for fuzzy matching parameters."""
    title_threshold_accept: int = 90      # accept if composite score >= this
    title_threshold_review: int = 80      # send to "needs review" if between review..accept
    author_weight: float = 0.25
    title_weight: float = 0.70
    year_bonus_per_match: int = 5         # bonus if publication_year matches exactly
    year_tolerance: int = 2               # within +/- 2 years gets a small bonus
    year_bonus_close: int = 2
    max_candidates_per_block: int = 50    # safety cap
    block_title_prefix: int = 14          # size of normalized title prefix for blocking
    top_k: int = 3                        # compute up to K top candidates for auditing


CFG = MatchConfig()


# ----------------------------
# Normalization utilities
# ----------------------------
PUNCT_TABLE = str.maketrans("", "", string.punctuation)

SERIES_PAT = re.compile(
    r"\((?:book|books|series|volume|vol\.?|part|#)\s*[\w\s\.\-#]*\)",
    flags=re.IGNORECASE
)
SUBTITLE_PAT = re.compile(r":\s+.*$")  # strip everything after ":" (subtitle)
EXTRA_SPACES = re.compile(r"\s+")


def normalize_whitespace(s: str) -> str:
    """Normalize whitespace in string."""
    return EXTRA_SPACES.sub(" ", s).strip()


def strip_series_markers(title: str) -> str:
    """Remove series markers from title."""
    return SERIES_PAT.sub("", title)


def strip_subtitle(title: str) -> str:
    """Remove subtitle from title."""
    return SUBTITLE_PAT.sub("", title)


def base_clean(text: str) -> str:
    """Basic text cleaning: accent folding, punctuation removal, lowercase."""
    text = unidecode(str(text))  # accent fold
    text = text.replace("'", "'")
    text = text.translate(PUNCT_TABLE)
    text = text.lower()
    text = normalize_whitespace(text)
    return text


def normalize_title(title: str) -> str:
    """Normalize title for matching."""
    title = strip_series_markers(str(title))
    title = strip_subtitle(title)
    title = base_clean(title)
    # Remove common suffix like "a novel"
    title = re.sub(r"\b(a|the)\s+novel\b", "", title)
    title = normalize_whitespace(title)
    return title


def parse_primary_author(authors_field: str) -> str:
    """
    Goodreads authors often "A; B; C" or "A, B".
    We take the first author.
    """
    if not isinstance(authors_field, str):
        return ""
    s = unidecode(authors_field).strip()
    # split on ';' or '&' or ' and ' or ','
    parts = re.split(r"\s*;|,| & |\sand\s|\s&\s", s, flags=re.IGNORECASE)
    return parts[0].strip() if parts else s


def normalize_author(author: str) -> str:
    """Normalize author name for matching."""
    author = base_clean(author)
    author = re.sub(r"\b(dr|mr|mrs|ms|prof)\b\.?", "", author)
    author = normalize_whitespace(author)
    return author


def author_last_name(author: str) -> str:
    """
    Heuristic: last token after cleaning as last name.
    """
    tokens = author.split()
    return tokens[-1] if tokens else ""


# ----------------------------
# Blocking keys
# ----------------------------
def title_block_key(title_norm: str, prefix_len: int) -> str:
    """Create blocking key from normalized title."""
    # Only alnum to make block stable
    alnum = re.sub(r"[^a-z0-9]+", "", title_norm)
    return alnum[:prefix_len]


def author_block_key(author_norm: str) -> str:
    """Create blocking key from normalized author."""
    # Use last name for block
    return author_last_name(author_norm)[:6]


# ----------------------------
# Composite score
# ----------------------------
def composite_score(
    title_score: float,
    author_score: float,
    pubyear_gr: Optional[float],
    pubyear_tx: Optional[float],
    cfg: MatchConfig = CFG
) -> float:
    """Calculate composite matching score."""
    score = cfg.title_weight * title_score + cfg.author_weight * author_score

    # Year bonus
    if pd.notna(pubyear_gr) and pd.notna(pubyear_tx):
        try:
            y1, y2 = int(pubyear_gr), int(pubyear_tx)
            if y1 == y2:
                score += cfg.year_bonus_per_match
            elif abs(y1 - y2) <= cfg.year_tolerance:
                score += cfg.year_bonus_close
        except Exception:
            pass

    # clip to [0, 100 + bonuses], then cap at 100
    return float(min(100.0, score))


# ----------------------------
# Matching routine
# ----------------------------
def build_blocks(df: pd.DataFrame, title_col: str, author_col: str, cfg: MatchConfig):
    """
    Adds normalized fields and block keys for efficient candidate generation.
    """
    df = df.copy()
    df["title_norm"] = df[title_col].fillna("").apply(normalize_title)
    df["author_primary"] = df[author_col].fillna("").apply(parse_primary_author)
    df["author_norm"] = df["author_primary"].apply(normalize_author)

    df["title_block"] = df["title_norm"].apply(lambda t: title_block_key(t, cfg.block_title_prefix))
    df["author_block"] = df["author_norm"].apply(author_block_key)
    return df


def candidate_pairs_for_block(block_df_left: pd.DataFrame, block_df_right: pd.DataFrame, cfg: MatchConfig):
    """
    For a given block (same title_block + author_block), produce top K candidate matches per left item.
    """
    candidates = []

    # Pre-collect right-side lists for RapidFuzz
    right_titles = block_df_right["title_norm"].tolist()
    right_authors = block_df_right["author_norm"].tolist()
    right_ids = block_df_right.index.tolist()
    right_pubyears = block_df_right.get("publication_year", pd.Series(index=block_df_right.index, dtype="float64"))

    for idx, row in block_df_left.iterrows():
        t_left = row["title_norm"]
        a_left = row["author_norm"]
        y_left = row.get("publication_year", None)

        # Title similarity against right block
        # Use token_set_ratio to be robust to word order and missing subtitles
        scored = []
        # Efficient: do a quick small-k candidate selection by title, then refine with author
        title_results = process.extract(
            t_left,
            right_titles,
            scorer=fuzz.token_set_ratio,
            limit=min(cfg.max_candidates_per_block, len(right_titles)),
        )

        for (cand_title, tscore, pos) in title_results:
            a_right = right_authors[pos]
            y_right = right_pubyears.iloc[pos] if isinstance(right_pubyears, pd.Series) else None

            # Author score
            ascore = fuzz.token_set_ratio(a_left, a_right)

            # Composite
            cscore = composite_score(tscore, ascore, y_left, y_right, cfg)

            candidates.append({
                "left_idx": idx,
                "right_idx": right_ids[pos],
                "title_score": tscore,
                "author_score": ascore,
                "composite": cscore,
            })

    if not candidates:
        return pd.DataFrame(columns=["left_idx","right_idx","title_score","author_score","composite"])

    cand_df = pd.DataFrame(candidates)
    # Keep top_k per left_idx
    cand_df = cand_df.sort_values(["left_idx", "composite"], ascending=[True, False])
    cand_df = cand_df.groupby("left_idx").head(CFG.top_k).reset_index(drop=True)
    return cand_df


def match_goodreads_to_texts(
    gr_df: pd.DataFrame,
    tx_df: pd.DataFrame,
    cfg: MatchConfig = CFG,
    gr_cols: Tuple[str, str, str] = ("title", "author_name", "publication_year"),
    tx_cols: Tuple[str, str, str] = ("title", "author", "publication_year"),
    gr_id_col: str = "work_id",
    tx_id_col: str = "text_book_id"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (matches_df, needs_review_df).
    - matches_df: one best match per Goodreads book with composite >= accept threshold
    - needs_review_df: top candidates with composite in [review, accept)
    """

    gr_title, gr_author, gr_year = gr_cols
    tx_title, tx_author, tx_year = tx_cols

    # Prepare / block
    gr_b = build_blocks(gr_df.set_index(gr_id_col), gr_title, gr_author, cfg)
    tx_b = build_blocks(tx_df.set_index(tx_id_col), tx_title, tx_author, cfg)

    # Inner blocking join
    # We join on both (title_block, author_block). Add a fallback looser block if needed.
    tx_cols_to_merge = [tx_id_col, "title_block", "author_block", "title_norm", "author_norm"]
    if tx_year is not None and tx_year in tx_b.columns:
        tx_cols_to_merge.append(tx_year)
    
    merged_blocks = gr_b.reset_index().merge(
        tx_b.reset_index()[tx_cols_to_merge],
        on=["title_block", "author_block"],
        how="inner",
        suffixes=("_gr", "_tx")
    )

    # If block is too small/large, we still process in groups
    if merged_blocks.empty:
        warnings.warn("No blocks found on strict keys. Consider loosening blocking (e.g., shorter title prefix).")

    # Group by block keys to limit pairwise comparisons
    all_candidates = []
    for (tb, ab), group in merged_blocks.groupby(["title_block", "author_block"]):
        left_cols = ["title_norm_gr", "author_norm_gr"]
        left_rename = {"title_norm_gr": "title_norm", "author_norm_gr": "author_norm"}
        if gr_year is not None and gr_year in group.columns:
            left_cols.append(gr_year)
            left_rename[gr_year] = "publication_year"
        
        right_cols = ["title_norm_tx", "author_norm_tx"]
        right_rename = {"title_norm_tx": "title_norm", "author_norm_tx": "author_norm"}
        if tx_year is not None and tx_year in group.columns:
            right_cols.append(tx_year)
            right_rename[tx_year] = "publication_year"
        
        left = group.drop_duplicates(subset=[gr_id_col]).set_index(gr_id_col)[left_cols]
        left = left.rename(columns=left_rename)
        right = group.drop_duplicates(subset=[tx_id_col]).set_index(tx_id_col)[right_cols]
        right = right.rename(columns=right_rename)

        cand_df = candidate_pairs_for_block(left, right, cfg)
        if not cand_df.empty:
            all_candidates.append(cand_df)

    if not all_candidates:
        return pd.DataFrame(), pd.DataFrame()

    cands = pd.concat(all_candidates, ignore_index=True)

    # Attach original info to candidates for sorting/tie-breaks and export
    gr_cols_to_merge = [gr_id_col, gr_title, gr_author, "title_norm", "author_norm"]
    if gr_year is not None and gr_year in gr_b.columns:
        gr_cols_to_merge.append(gr_year)
    
    tx_cols_to_merge = [tx_id_col, tx_title, tx_author, "title_norm", "author_norm"]
    if tx_year is not None and tx_year in tx_b.columns:
        tx_cols_to_merge.append(tx_year)
    
    cands = (
        cands
        .merge(gr_b.reset_index()[gr_cols_to_merge],
               left_on="left_idx", right_on=gr_id_col, how="left", suffixes=("","_gr"))
        .merge(tx_b.reset_index()[tx_cols_to_merge],
               left_on="right_idx", right_on=tx_id_col, how="left", suffixes=("","_tx"))
    )

    # Year proximity for tie-break
    def year_dist(row):
        y1, y2 = None, None
        if gr_year is not None and gr_year in row:
            y1 = row.get(gr_year)
        if tx_year is not None and tx_year in row:
            y2 = row.get(tx_year)
        try:
            if pd.notna(y1) and pd.notna(y2):
                return abs(int(y1) - int(y2))
        except Exception:
            pass
        return 999

    cands["year_distance"] = cands.apply(year_dist, axis=1)

    # Length difference in titles (helps pick closer variants)
    cands["title_len_diff"] = (cands["title_norm"].str.len() - cands["title_norm_tx"].str.len()).abs()

    # Choose best single match per Goodreads book
    best = (
        cands.sort_values(
            ["left_idx", "composite", "title_score", "author_score", "year_distance", "title_len_diff"],
            ascending=[True, False, False, False, True, True]
        )
        .groupby("left_idx", as_index=False)
        .first()
    )

    # Split into accept vs review
    accept_mask = best["composite"] >= cfg.title_threshold_accept
    review_mask = (best["composite"] >= cfg.title_threshold_review) & (~accept_mask)

    matches = best[accept_mask].copy()
    needs_review = best[review_mask].copy()

    # Also surface the full top-k candidate list for those needing review
    # (handy if you want to build a UI to pick alternates)
    if not needs_review.empty:
        topk_for_review = (
            cands[cands["left_idx"].isin(needs_review["left_idx"])]
            .sort_values(["left_idx", "composite"], ascending=[True, False])
        )
    else:
        topk_for_review = pd.DataFrame()

    # Pretty columns
    def rename_cols(df):
        return df.rename(columns={
            "left_idx": "goodreads_work_id",
            "right_idx": "text_book_id",
            gr_title: "gr_title",
            gr_author: "gr_author_name",
            gr_year: "gr_publication_year",
            tx_title: "tx_title",
            tx_author: "tx_author",
            tx_year: "tx_publication_year"
        })

    return rename_cols(matches), rename_cols(topk_for_review)


# ----------------------------
# Text extraction utilities
# ----------------------------
def extract_title_author_from_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract title and author from text field for datasets like AlekseyKorshuk/romance-books.
    Assumes format: "Author Name Title Name\n\nText content..."
    """
    df = df.copy()
    
    def extract_metadata(text):
        if not isinstance(text, str):
            return "", ""
        
        # Split by lines and clean
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
        
        if len(lines) < 2:
            return "", ""
        
        # Look for patterns like "Author Name Title Name" in first line
        first_line = lines[0]
        
        # Try to split on common patterns
        # Pattern 1: "Author Name Title Name" (no clear separator)
        # Pattern 2: "Author Name by Title Name" 
        # Pattern 3: "Title Name by Author Name"
        
        # Check for "by" pattern
        if ' by ' in first_line.lower():
            parts = first_line.split(' by ', 1)
            if len(parts) == 2:
                # Could be "Title by Author" or "Author by Title"
                # Assume first part is author, second is title
                author = parts[0].strip()
                title = parts[1].strip()
            else:
                author = parts[0].strip()
                title = ""
        else:
            # No clear separator, try to split on word boundaries
            # Look for common title patterns
            words = first_line.split()
            if len(words) >= 2:
                # Heuristic: if first word is capitalized and second is too, 
                # likely "Author Name Title Name"
                if len(words) >= 4:
                    # Try splitting in the middle
                    mid = len(words) // 2
                    author = ' '.join(words[:mid])
                    title = ' '.join(words[mid:])
                else:
                    # Short line, assume first word is author, rest is title
                    author = words[0]
                    title = ' '.join(words[1:])
            else:
                author = first_line
                title = ""
        
        # Clean up common patterns
        if title.startswith('by '):
            title = title[3:]
        if author.startswith('by '):
            author = author[3:]
            
        # Remove common prefixes/suffixes
        author = author.replace('Author:', '').replace('Author', '').strip()
        title = title.replace('Title:', '').replace('Title', '').strip()
        
        # Filter out very short or meaningless titles/authors
        if len(author) < 2 or len(title) < 2:
            return "", ""
        if author.lower() in ['unknown', 'anonymous', 'n/a']:
            return "", ""
        if title.lower() in ['unknown', 'untitled', 'n/a']:
            return "", ""
            
        return author, title
    
    # Extract author and title
    metadata = df['text'].apply(extract_metadata)
    df['author'] = [m[0] for m in metadata]
    df['title'] = [m[1] for m in metadata]
    
    # Filter out rows where we couldn't extract meaningful metadata
    df = df[(df['author'] != '') & (df['title'] != '')]
    
    return df


# ----------------------------
# Example main
# ----------------------------
def main(
    goodreads_csv: str,
    output_matches_csv: str = "matches_definitive.csv",
    output_review_csv: str = "matches_needs_review.csv",
    use_hf_texts: bool = True,
    local_texts_csv: Optional[str] = None
):
    """Main function to run the matching process."""
    # ---- Load Goodreads subset
    print(f"Loading Goodreads dataset from: {goodreads_csv}")
    gr = pd.read_csv(goodreads_csv)

    # Expect at least: work_id, title, author_name, publication_year
    required_gr = {"work_id", "title", "author_name"}
    missing_gr = required_gr - set(gr.columns)
    if missing_gr:
        raise ValueError(f"Goodreads CSV missing columns: {missing_gr}")

    print(f"Loaded {len(gr)} Goodreads books")

    # ---- Load text dataset (Hugging Face OR local CSV)
    if use_hf_texts:
        if not HF_AVAILABLE:
            raise RuntimeError("Hugging Face 'datasets' not available. Install or use local_texts_csv.")
        print("Loading AlekseyKorshuk/romance-books from HF...")
        ds = load_dataset("AlekseyKorshuk/romance-books")
        tx = ds["train"].to_pandas()
        # Try to infer reasonable columns
        # You may need to adjust these depending on the dataset schema
        # Typical columns: 'title', 'author', 'year' or 'publication_year', 'text' ...
        # We'll create a stable ID if missing
        if "id" in tx.columns:
            tx["text_book_id"] = tx["id"]
        else:
            tx["text_book_id"] = tx.index.astype(str)

        # Harmonize year column name if needed
        if "year" in tx.columns and "publication_year" not in tx.columns:
            tx = tx.rename(columns={"year": "publication_year"})

        # Handle datasets where title/author are embedded in text (like AlekseyKorshuk/romance-books)
        if "title" not in tx.columns or "author" not in tx.columns:
            if "text" in tx.columns:
                print("Extracting title and author from text field...")
                tx = extract_title_author_from_text(tx)
            else:
                raise ValueError("Text dataset missing 'title' or 'author' columns. Inspect and map column names.")
    else:
        if not local_texts_csv:
            raise ValueError("local_texts_csv must be provided if use_hf_texts=False")
        print(f"Loading local text dataset from: {local_texts_csv}")
        tx = pd.read_csv(local_texts_csv)
        if "text_book_id" not in tx.columns:
            tx["text_book_id"] = tx.index.astype(str)

    print(f"Loaded {len(tx)} text books")

    # ---- Run matching
    print("Running fuzzy matching...")
    matches, needs_review = match_goodreads_to_texts(
        gr_df=gr,
        tx_df=tx,
        gr_cols=("title", "author_name", "publication_year" if "publication_year" in gr.columns else None),
        tx_cols=("title", "author", "publication_year" if "publication_year" in tx.columns else None),
        gr_id_col="work_id",
        tx_id_col="text_book_id"
    )

    # ---- Save
    if not matches.empty:
        matches.to_csv(output_matches_csv, index=False)
        print(f"Saved definitive matches: {output_matches_csv} (n={len(matches)})")
    else:
        print("No definitive matches above accept threshold.")

    if not needs_review.empty:
        needs_review.to_csv(output_review_csv, index=False)
        print(f"Saved candidates needing review: {output_review_csv} (rows={len(needs_review)})")
    else:
        print("No items in review band.")

    # Print summary
    print(f"\nMatching Summary:")
    print(f"- Total Goodreads books: {len(gr)}")
    print(f"- Total text books: {len(tx)}")
    print(f"- Definitive matches: {len(matches)}")
    print(f"- Needs review: {len(needs_review)}")
    print(f"- Match rate: {len(matches) / len(gr) * 100:.1f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fuzzy match Goodreads to text corpus by title–author.")
    parser.add_argument("--goodreads_csv", type=str, required=True, help="Path to Goodreads subset CSV")
    parser.add_argument("--use_hf_texts", action="store_true", help="If set, use HF AlekseyKorshuk/romance-books")
    parser.add_argument("--local_texts_csv", type=str, default=None, help="Alternative: local CSV with texts metadata")
    parser.add_argument("--out_matches", type=str, default="matches_definitive.csv")
    parser.add_argument("--out_review", type=str, default="matches_needs_review.csv")
    args = parser.parse_args()

    main(
        goodreads_csv=args.goodreads_csv,
        output_matches_csv=args.out_matches,
        output_review_csv=args.out_review,
        use_hf_texts=args.use_hf_texts,
        local_texts_csv=args.local_texts_csv
    )

