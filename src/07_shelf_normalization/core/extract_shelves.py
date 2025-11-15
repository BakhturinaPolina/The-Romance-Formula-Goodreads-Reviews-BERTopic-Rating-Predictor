#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract and count shelves from main dataset for testing simple cleaner.

Usage:
    python extract_shelves.py --input ../../data/processed/main_dataset.csv --output shelf_canonical_test.csv
"""

import argparse
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_shelves_from_csv(csv_path: Path, shelves_col: str = "shelves_str") -> pd.DataFrame:
    """Extract unique shelves and their counts from CSV."""
    logger.info(f"Loading {csv_path}")
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    
    if shelves_col not in df.columns:
        raise ValueError(f"Column '{shelves_col}' not found. Available columns: {list(df.columns)[:10]}...")
    
    logger.info(f"Found {len(df):,} rows")
    logger.info(f"Extracting shelves from column '{shelves_col}'")
    
    # Extract all shelves
    shelf_counts = Counter()
    for idx, row in df.iterrows():
        if pd.notna(row[shelves_col]) and str(row[shelves_col]).strip():
            shelves_str = str(row[shelves_col]).strip()
            # Split by comma
            shelves = [s.strip() for s in shelves_str.split(',') if s.strip()]
            for shelf in shelves:
                shelf_counts[shelf] += 1
        
        if (idx + 1) % 10000 == 0:
            logger.info(f"  Processed {idx + 1:,} rows, found {len(shelf_counts):,} unique shelves so far")
    
    logger.info(f"Extracted {len(shelf_counts):,} unique shelves")
    
    # Create DataFrame
    canon_df = pd.DataFrame([
        {
            'shelf_raw': shelf,
            'shelf_canon': shelf.lower().strip(),  # Simple canonicalization
            'count': count
        }
        for shelf, count in shelf_counts.items()
    ])
    
    return canon_df.sort_values('count', ascending=False)


def main():
    parser = argparse.ArgumentParser(description="Extract shelves from main dataset")
    parser.add_argument('--input', type=Path, required=True, help='Input CSV file')
    parser.add_argument('--output', type=Path, required=True, help='Output shelf_canonical.csv')
    parser.add_argument('--shelves-col', default='shelves_str', help='Column name with shelves (default: shelves_str)')
    
    args = parser.parse_args()
    
    canon_df = extract_shelves_from_csv(args.input, args.shelves_col)
    
    logger.info(f"Saving to {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canon_df.to_csv(args.output, index=False)
    
    logger.info("=" * 60)
    logger.info("EXTRACTION SUMMARY:")
    logger.info(f"  Total unique shelves: {len(canon_df):,}")
    logger.info(f"  Total occurrences: {canon_df['count'].sum():,}")
    logger.info(f"  Top 20 shelves:")
    for idx, (_, row) in enumerate(canon_df.head(20).iterrows(), 1):
        logger.info(f"    {idx:2d}. '{row['shelf_canon']}' (count={row['count']:,})")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

