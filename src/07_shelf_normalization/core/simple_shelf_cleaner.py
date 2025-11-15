#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Shelf Cleaner - Easy improvements for cleaner shelf names.

This module implements the simplest possible improvements to make shelf names
cleaner and more useful for topic modeling:
1. Plural/singular normalization
2. Generic stopword removal
3. Common suffix/prefix removal
4. Frequency-based filtering

Usage:
    python simple_shelf_cleaner.py --input shelf_canonical.csv --output shelf_canonical_cleaned.csv
"""

from __future__ import annotations
import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Set, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Generic words that don't add thematic value
# Note: Don't include 'read' here as it's part of "to-read", "currently-reading" etc.
GENERIC_STOPWORDS: Set[str] = {
    'book', 'books', 'novel', 'novels', 'fiction', 'story', 'stories'
}

# Common suffixes/prefixes to remove (but NOT "-read" as it's part of "to-read")
# Only remove if the remaining part is meaningful
COMMON_SUFFIXES: Set[str] = {'-books', '-novels', '-fiction', '-story', '-stories'}
COMMON_PREFIXES: Set[str] = {'books-', 'novels-', 'fiction-', 'story-'}

# Patterns to preserve (don't modify these)
PRESERVE_PATTERNS: Set[str] = {'to-read', 'currently-reading', 'want-to-read', 'wtr', 'tbr'}


def normalize_plural(shelf: str, shelf_counts: Counter) -> str:
    """
    Simple plural normalization - only merge if both forms exist.
    
    Args:
        shelf: Shelf name to normalize
        shelf_counts: Counter of shelf frequencies
        
    Returns:
        Normalized shelf (singular form if it exists and is more common)
    """
    if not shelf or len(shelf) <= 3:
        return shelf
    
    # Try to find singular form
    singular = None
    
    if shelf.endswith('ies') and len(shelf) > 4:
        # "fantasies" -> "fantasy"
        singular = shelf[:-3] + 'y'
    elif shelf.endswith('es') and len(shelf) > 4:
        # "romances" -> "romance"
        singular = shelf[:-2]
    elif shelf.endswith('s') and len(shelf) > 2:
        # "books" -> "book" (but we'll filter "book" anyway)
        singular = shelf[:-1]
    
    # Use singular if it exists and is at least as common
    if singular and singular in shelf_counts:
        if shelf_counts[singular] >= shelf_counts[shelf]:
            return singular
    
    return shelf


def remove_generic_stopwords(shelf: str) -> str:
    """
    Remove generic words that don't add thematic value.
    Handles both space-separated and hyphenated words.
    
    Args:
        shelf: Shelf name
        
    Returns:
        Shelf with generic words removed
    """
    if not shelf:
        return shelf
    
    original = shelf
    
    # Handle hyphenated words: split by both space and hyphen
    # First try space-separated
    words = shelf.split()
    if len(words) > 1:
        filtered = [w for w in words if w not in GENERIC_STOPWORDS]
        result = ' '.join(filtered).strip()
        if result and result != original:
            return result
    
    # Handle hyphenated: split by hyphen
    parts = shelf.split('-')
    if len(parts) > 1:
        filtered = [p for p in parts if p not in GENERIC_STOPWORDS]
        if len(filtered) > 0:
            result = '-'.join(filtered).strip('-')
            # Don't return if result is too short or just punctuation
            if result and len(result) > 2 and result != original:
                return result
    
    # Keep original if no changes
    return original


def clean_suffixes_prefixes(shelf: str) -> str:
    """
    Remove common generic suffixes/prefixes.
    Preserves important patterns like "to-read".
    
    Args:
        shelf: Shelf name
        
    Returns:
        Shelf with suffixes/prefixes removed
    """
    if not shelf:
        return shelf
    
    # Preserve important patterns
    if shelf in PRESERVE_PATTERNS:
        return shelf
    
    original = shelf
    
    # Remove suffixes (but check if result is meaningful)
    for suffix in COMMON_SUFFIXES:
        if shelf.endswith(suffix):
            cleaned = shelf[:-len(suffix)].strip('-').strip()
            # Only apply if cleaned result is meaningful (not too short, not just punctuation)
            if cleaned and len(cleaned) > 2 and cleaned not in PRESERVE_PATTERNS:
                shelf = cleaned
                break
    
    # Remove prefixes (but check if result is meaningful)
    for prefix in COMMON_PREFIXES:
        if shelf.startswith(prefix):
            cleaned = shelf[len(prefix):].strip('-').strip()
            # Only apply if cleaned result is meaningful
            if cleaned and len(cleaned) > 2 and cleaned not in PRESERVE_PATTERNS:
                shelf = cleaned
                break
    
    # Return cleaned version if non-empty and meaningful, else original
    return shelf if shelf and len(shelf) > 1 else original


def apply_simple_cleaning(
    canon_df: pd.DataFrame,
    min_frequency: int = 3,
    apply_plural: bool = True,
    apply_stopwords: bool = True,
    apply_suffixes: bool = True,
    verbose: bool = False
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Apply simple cleaning rules to canonical shelves.
    
    Args:
        canon_df: DataFrame with 'shelf_canon' and 'count' columns
        min_frequency: Minimum frequency to keep a shelf
        apply_plural: Whether to normalize plurals
        apply_stopwords: Whether to remove stopwords
        apply_suffixes: Whether to remove suffixes/prefixes
        verbose: Whether to print detailed progress
        
    Returns:
        Tuple of (cleaned DataFrame, statistics dict)
    """
    stats = {
        'original_count': len(canon_df),
        'after_frequency_filter': 0,
        'after_plural_norm': 0,
        'after_stopwords': 0,
        'after_suffixes': 0,
        'final_count': 0,
        'plural_merges': [],
        'stopword_removals': [],
        'suffix_removals': [],
        'frequency_filtered': []
    }
    
    # Build frequency counter
    if 'count' in canon_df.columns:
        shelf_counts = Counter(dict(zip(canon_df['shelf_canon'], canon_df['count'])))
    else:
        shelf_counts = Counter(canon_df['shelf_canon'].value_counts().to_dict())
    
    # Step 1: Frequency filtering
    if min_frequency > 1:
        filtered_out = canon_df[canon_df['count'] < min_frequency].copy()
        stats['frequency_filtered'] = filtered_out[['shelf_canon', 'count']].to_dict('records')[:50]  # Sample
        canon_df = canon_df[canon_df['count'] >= min_frequency].copy()
        stats['after_frequency_filter'] = len(canon_df)
        logger.info(f"After frequency filter (min={min_frequency}): {len(canon_df):,} shelves")
        if verbose:
            logger.info(f"  Filtered out {len(filtered_out):,} rare shelves (showing top 20):")
            for _, row in filtered_out.nlargest(20, 'count').iterrows():
                logger.info(f"    - '{row['shelf_canon']}' (count={row['count']})")
    else:
        stats['after_frequency_filter'] = len(canon_df)
    
    # Step 2: Plural normalization
    if apply_plural:
        canon_df['shelf_cleaned'] = canon_df['shelf_canon'].apply(
            lambda s: normalize_plural(s, shelf_counts)
        )
        # Track merges
        plural_merges = canon_df[canon_df['shelf_canon'] != canon_df['shelf_cleaned']].copy()
        if verbose and len(plural_merges) > 0:
            logger.info(f"  Plural normalization found {len(plural_merges):,} potential merges (showing top 20):")
            for _, row in plural_merges.head(20).iterrows():
                logger.info(f"    - '{row['shelf_canon']}' → '{row['shelf_cleaned']}' (count={row['count']})")
        stats['plural_merges'] = plural_merges[['shelf_canon', 'shelf_cleaned', 'count']].to_dict('records')[:50]
        # Merge duplicates created by plural normalization
        canon_df = canon_df.groupby('shelf_cleaned', as_index=False).agg({
            'shelf_canon': 'first',  # Keep first original
            'count': 'sum'  # Sum frequencies
        })
        canon_df['shelf_canon'] = canon_df['shelf_cleaned']
        canon_df = canon_df.drop(columns=['shelf_cleaned'])
        stats['after_plural_norm'] = len(canon_df)
        logger.info(f"After plural normalization: {len(canon_df):,} shelves (merged {len(plural_merges):,} plurals)")
    else:
        stats['after_plural_norm'] = len(canon_df)
    
    # Step 3: Stopword removal
    if apply_stopwords:
        canon_df['shelf_cleaned'] = canon_df['shelf_canon'].apply(remove_generic_stopwords)
        # Track removals
        stopword_changes = canon_df[canon_df['shelf_canon'] != canon_df['shelf_cleaned']].copy()
        if verbose and len(stopword_changes) > 0:
            logger.info(f"  Stopword removal changed {len(stopword_changes):,} shelves (showing top 20):")
            for _, row in stopword_changes.head(20).iterrows():
                logger.info(f"    - '{row['shelf_canon']}' → '{row['shelf_cleaned']}' (count={row['count']})")
        stats['stopword_removals'] = stopword_changes[['shelf_canon', 'shelf_cleaned', 'count']].to_dict('records')[:50]
        # Merge duplicates
        canon_df = canon_df.groupby('shelf_cleaned', as_index=False).agg({
            'shelf_canon': 'first',
            'count': 'sum'
        })
        canon_df['shelf_canon'] = canon_df['shelf_cleaned']
        canon_df = canon_df.drop(columns=['shelf_cleaned'])
        stats['after_stopwords'] = len(canon_df)
        logger.info(f"After stopword removal: {len(canon_df):,} shelves (changed {len(stopword_changes):,})")
    else:
        stats['after_stopwords'] = len(canon_df)
    
    # Step 4: Suffix/prefix removal
    if apply_suffixes:
        canon_df['shelf_cleaned'] = canon_df['shelf_canon'].apply(clean_suffixes_prefixes)
        # Track removals
        suffix_changes = canon_df[canon_df['shelf_canon'] != canon_df['shelf_cleaned']].copy()
        if verbose and len(suffix_changes) > 0:
            logger.info(f"  Suffix/prefix removal changed {len(suffix_changes):,} shelves (showing top 20):")
            for _, row in suffix_changes.head(20).iterrows():
                logger.info(f"    - '{row['shelf_canon']}' → '{row['shelf_cleaned']}' (count={row['count']})")
        stats['suffix_removals'] = suffix_changes[['shelf_canon', 'shelf_cleaned', 'count']].to_dict('records')[:50]
        # Merge duplicates
        canon_df = canon_df.groupby('shelf_cleaned', as_index=False).agg({
            'shelf_canon': 'first',
            'count': 'sum'
        })
        canon_df['shelf_canon'] = canon_df['shelf_cleaned']
        canon_df = canon_df.drop(columns=['shelf_cleaned'])
        stats['after_suffixes'] = len(canon_df)
        logger.info(f"After suffix/prefix removal: {len(canon_df):,} shelves (changed {len(suffix_changes):,})")
    else:
        stats['after_suffixes'] = len(canon_df)
    
    stats['final_count'] = len(canon_df)
    
    return canon_df, stats


def main():
    parser = argparse.ArgumentParser(
        description="Apply simple cleaning rules to shelf canonicalization"
    )
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input shelf_canonical.csv file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output cleaned shelf_canonical.csv file'
    )
    parser.add_argument(
        '--min-frequency',
        type=int,
        default=3,
        help='Minimum frequency to keep a shelf (default: 3)'
    )
    parser.add_argument(
        '--no-plural',
        action='store_true',
        help='Skip plural normalization'
    )
    parser.add_argument(
        '--no-stopwords',
        action='store_true',
        help='Skip stopword removal'
    )
    parser.add_argument(
        '--no-suffixes',
        action='store_true',
        help='Skip suffix/prefix removal'
    )
    parser.add_argument(
        '--stats',
        type=Path,
        help='Output statistics JSON file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress and examples'
    )
    
    args = parser.parse_args()
    
    # Load input
    logger.info(f"Loading {args.input}")
    canon_df = pd.read_csv(args.input, dtype=str)
    
    # Ensure required columns
    if 'shelf_canon' not in canon_df.columns:
        raise ValueError("Input must have 'shelf_canon' column")
    if 'count' not in canon_df.columns:
        # Try to infer from other columns or create
        if 'shelf_raw' in canon_df.columns:
            canon_df['count'] = canon_df.groupby('shelf_canon')['shelf_canon'].transform('count')
        else:
            canon_df['count'] = 1
    
    # Convert count to int
    canon_df['count'] = pd.to_numeric(canon_df['count'], errors='coerce').fillna(1).astype(int)
    
    # Print initial statistics
    logger.info("=" * 60)
    logger.info("INPUT STATISTICS:")
    logger.info(f"  Total shelves: {len(canon_df):,}")
    logger.info(f"  Total occurrences: {canon_df['count'].sum():,}")
    logger.info(f"  Unique shelves: {canon_df['shelf_canon'].nunique():,}")
    logger.info(f"  Min frequency: {canon_df['count'].min()}")
    logger.info(f"  Max frequency: {canon_df['count'].max()}")
    logger.info(f"  Mean frequency: {canon_df['count'].mean():.1f}")
    logger.info(f"  Median frequency: {canon_df['count'].median():.1f}")
    logger.info("")
    logger.info("  Top 20 most frequent shelves:")
    for idx, (shelf, count) in enumerate(canon_df.nlargest(20, 'count')[['shelf_canon', 'count']].values, 1):
        logger.info(f"    {idx:2d}. '{shelf}' (count={count:,})")
    logger.info("=" * 60)
    logger.info("")
    
    # Apply cleaning
    cleaned_df, stats = apply_simple_cleaning(
        canon_df,
        min_frequency=args.min_frequency,
        apply_plural=not args.no_plural,
        apply_stopwords=not args.no_stopwords,
        apply_suffixes=not args.no_suffixes,
        verbose=args.verbose
    )
    
    # Save output
    logger.info(f"Saving cleaned shelves to {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(args.output, index=False)
    
    # Save statistics
    if args.stats:
        import json
        with open(args.stats, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Statistics saved to {args.stats}")
    
    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("CLEANING SUMMARY:")
    logger.info("=" * 60)
    logger.info(f"  Original shelves: {stats['original_count']:,}")
    logger.info(f"  After frequency filter: {stats['after_frequency_filter']:,} (removed {stats['original_count'] - stats['after_frequency_filter']:,})")
    logger.info(f"  After plural norm: {stats['after_plural_norm']:,} (merged {stats['after_frequency_filter'] - stats['after_plural_norm']:,})")
    logger.info(f"  After stopwords: {stats['after_stopwords']:,} (merged {stats['after_plural_norm'] - stats['after_stopwords']:,})")
    logger.info(f"  After suffixes: {stats['after_suffixes']:,} (merged {stats['after_stopwords'] - stats['after_suffixes']:,})")
    logger.info(f"  Final shelves: {stats['final_count']:,}")
    logger.info(f"  Total reduction: {(1 - stats['final_count']/stats['original_count'])*100:.1f}%")
    logger.info("")
    logger.info("  Top 20 most frequent cleaned shelves:")
    for idx, (shelf, count) in enumerate(cleaned_df.nlargest(20, 'count')[['shelf_canon', 'count']].values, 1):
        logger.info(f"    {idx:2d}. '{shelf}' (count={count:,})")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

