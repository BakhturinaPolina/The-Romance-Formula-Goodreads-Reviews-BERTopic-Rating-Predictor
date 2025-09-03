#!/usr/bin/env python3
"""
Quick Exclusion Analysis for Romance Novel Dataset
Calculate how many books remain after excluding problematic records.
"""

import pandas as pd
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from data_quality.data_quality_assessment import DataQualityAssessment


def quick_exclusion_analysis():
    """Quick analysis of how many books remain after exclusions."""
    print("🔍 Quick Exclusion Analysis")
    print("=" * 50)
    
    # Initialize assessor and load data
    assessor = DataQualityAssessment()
    
    if not assessor.load_dataset():
        print("❌ Failed to load dataset")
        return
    
    data = assessor.data
    total_books = len(data)
    print(f"📚 Total books in dataset: {total_books:,}")
    print()
    
    # Track exclusions
    exclusions = {}
    
    # 1. Missing num_pages_median
    missing_pages = data['num_pages_median'].isna()
    missing_pages_count = missing_pages.sum()
    exclusions['missing_pages'] = missing_pages_count
    print(f"📏 Missing num_pages_median: {missing_pages_count:,} ({missing_pages_count/total_books*100:.1f}%)")
    
    # 2. Missing descriptions
    missing_desc = data['description'].isna() | (data['description'] == '')
    missing_desc_count = missing_desc.sum()
    exclusions['missing_descriptions'] = missing_desc_count
    print(f"📝 Missing descriptions: {missing_desc_count:,} ({missing_desc_count/total_books*100:.1f}%)")
    
    # 3. Very short descriptions (<50 chars)
    valid_descriptions = data['description'].notna() & (data['description'] != '')
    short_desc = valid_descriptions & (data['description'].str.len() < 50)
    short_desc_count = short_desc.sum()
    exclusions['short_descriptions'] = short_desc_count
    print(f"📝 Very short descriptions (<50 chars): {short_desc_count:,} ({short_desc_count/total_books*100:.1f}%)")
    
    # 4. Combined exclusions
    combined_exclusions = missing_pages | missing_desc | short_desc
    combined_exclusions_count = combined_exclusions.sum()
    exclusions['combined'] = combined_exclusions_count
    print(f"🚫 Combined exclusions: {combined_exclusions_count:,} ({combined_exclusions_count/total_books*100:.1f}%)")
    
    # 5. Books remaining
    remaining_books = total_books - combined_exclusions_count
    remaining_percentage = remaining_books / total_books * 100
    exclusions['remaining'] = remaining_books
    print(f"✅ Books remaining: {remaining_books:,} ({remaining_percentage:.1f}%)")
    
    print()
    print("📊 Summary:")
    print(f"  - Total books: {total_books:,}")
    print(f"  - Excluded: {combined_exclusions_count:,}")
    print(f"  - Remaining: {remaining_books:,} ({remaining_percentage:.1f}%)")
    
    # Show breakdown of exclusions
    print()
    print("🔍 Exclusion Breakdown:")
    print(f"  - Missing pages only: {missing_pages_count:,}")
    print(f"  - Missing descriptions only: {missing_desc_count:,}")
    print(f"  - Short descriptions only: {short_desc_count:,}")
    
    # Check overlap between exclusions
    pages_only = missing_pages & ~missing_desc & ~short_desc
    desc_only = ~missing_pages & missing_desc & ~short_desc
    short_only = ~missing_pages & ~missing_desc & short_desc
    
    print(f"  - Pages only: {pages_only.sum():,}")
    print(f"  - Descriptions only: {desc_only.sum():,}")
    print(f"  - Short descriptions only: {short_only.sum():,}")
    
    # Check combinations
    pages_and_desc = missing_pages & missing_desc
    pages_and_short = missing_pages & short_desc
    desc_and_short = missing_desc & short_desc
    all_three = missing_pages & missing_desc & short_desc
    
    print(f"  - Pages + Descriptions: {pages_and_desc.sum():,}")
    print(f"  - Pages + Short: {pages_and_short.sum():,}")
    print(f"  - Descriptions + Short: {desc_and_short.sum():,}")
    print(f"  - All three issues: {all_three.sum():,}")
    
    return exclusions


if __name__ == "__main__":
    quick_exclusion_analysis()
