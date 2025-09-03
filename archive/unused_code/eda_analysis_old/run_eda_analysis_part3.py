#!/usr/bin/env python3
"""
Run EDA Analysis Part 3: Publication Trends, Popularity, and Subgenre Analysis
Final part of the comprehensive EDA analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json
from pathlib import Path
from collections import Counter, defaultdict
import warnings

# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 1000)

def main():
    """Run the final part of EDA analysis."""
    print("🔍 FINAL EDA ANALYSIS - PART 3")
    print("=" * 60)
    
    # Load dataset
    print("📚 Loading dataset...")
    dataset_path = "data/processed/final_books_2000_2020_en_20250901_024106.csv"
    
    try:
        df = pd.read_csv(dataset_path)
        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Shape: {df.shape}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Step 12: Publication & Popularity Analysis
    print("\n🔍 PUBLICATION & POPULARITY ANALYSIS")
    print("=" * 50)
    
    # Publication year distribution
    year_counts = df['publication_year'].value_counts().sort_index()
    print(f"📅 Publication year distribution:")
    print(f"   - Range: {df['publication_year'].min()} - {df['publication_year'].max()}")
    print(f"   - Most common year: {year_counts.idxmax()} ({year_counts.max():,} books)")
    print(f"   - Least common year: {year_counts.idxmin()} ({year_counts.min():,} books)")
    
    # Popularity metrics
    print(f"\n⭐ Popularity metrics:")
    print(f"   - Average rating (mean): {df['average_rating_weighted_mean'].mean():.2f}")
    print(f"   - Average rating (median): {df['average_rating_weighted_mean'].median():.2f}")
    print(f"   - Ratings count (mean): {df['ratings_count_sum'].mean():,.0f}")
    print(f"   - Ratings count (median): {df['ratings_count_sum'].median():,.0f}")
    print(f"   - Reviews count (mean): {df['text_reviews_count_sum'].mean():,.0f}")
    print(f"   - Reviews count (median): {df['text_reviews_count_sum'].median():,.0f}")
    
    # Step 13: Subgenre Signal Analysis
    print("\n🔍 SUBGENRE SIGNAL ANALYSIS")
    print("=" * 50)
    
    # Analyze popular shelves for subgenre signals
    print("📚 Popular shelves analysis:")
    
    # Sample some popular shelves to understand structure
    sample_shelves = df['popular_shelves'].dropna().head(10)
    print(f"\n📋 Sample popular shelves:")
    for i, shelves in enumerate(sample_shelves):
        try:
            shelves_list = json.loads(shelves)
            print(f"  {i+1}. {shelves_list[:5]}...")  # Show first 5 shelves
        except:
            print(f"  {i+1}. {shelves[:100]}...")  # Show first 100 chars if not JSON
    
    # Check if popular_shelves is JSON format
    json_format_count = 0
    for shelves in df['popular_shelves'].dropna():
        try:
            json.loads(shelves)
            json_format_count += 1
        except:
            pass
    
    print(f"\n📊 Popular shelves format:")
    print(f"   - JSON format: {json_format_count:,} ({json_format_count/len(df['popular_shelves'].dropna())*100:.1f}%)")
    print(f"   - Non-JSON format: {len(df['popular_shelves'].dropna()) - json_format_count:,}")
    
    # Step 14: Subgenre Extraction from Popular Shelves
    print("\n🔍 SUBGENRE EXTRACTION FROM POPULAR SHELVES")
    print("=" * 50)
    
    # Target subgenres for research
    target_subgenres = [
        'contemporary romance', 'historical romance', 'paranormal romance',
        'romantic suspense', 'romantic fantasy', 'science fiction romance'
    ]
    
    # Extract subgenre signals
    subgenre_counts = defaultdict(int)
    subgenre_examples = defaultdict(list)
    
    for shelves in df['popular_shelves'].dropna():
        try:
            shelves_list = json.loads(shelves)
            for shelf in shelves_list:
                shelf_lower = shelf.lower()
                for subgenre in target_subgenres:
                    if subgenre in shelf_lower:
                        subgenre_counts[subgenre] += 1
                        # Store example book title
                        if len(subgenre_examples[subgenre]) < 3:
                            book_idx = df[df['popular_shelves'] == shelves].index[0]
                            subgenre_examples[subgenre].append(df.loc[book_idx, 'title'])
        except:
            continue
    
    print(f"📊 Subgenre signals found:")
    for subgenre, count in sorted(subgenre_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(df)) * 100
        print(f"   - {subgenre}: {count:,} books ({percentage:.1f}%)")
        if subgenre_examples[subgenre]:
            print(f"     Examples: {', '.join(subgenre_examples[subgenre])}")
    
    # Step 15: Comprehensive Cleaning Recommendations
    print("\n🔍 COMPREHENSIVE CLEANING RECOMMENDATIONS")
    print("=" * 50)
    
    print("📋 SUMMARY OF FINDINGS:")
    print(f"   - Dataset size: {len(df):,} romance novels")
    print(f"   - Publication range: {df['publication_year'].min()} - {df['publication_year'].max()}")
    print(f"   - Series coverage: {df['series_id'].notna().sum():,} books ({df['series_id'].notna().sum()/len(df)*100:.1f}%)")
    print(f"   - Missing descriptions: {df['description'].isnull().sum():,} ({df['description'].isnull().sum()/len(df)*100:.1f}%)")
    print(f"   - HTML artifacts: {df['description'].str.contains(r'<[^>]+>', regex=True, na=False).sum():,} descriptions")
    print(f"   - Author variations: {len([name for name, ids in defaultdict(list).items() if len(ids) > 1]):,} potential duplicates")
    
    print("\n🧹 RECOMMENDED CLEANING STEPS:")
    print("\n1. TITLE CLEANING:")
    print("   - Extract series numbers and prefixes (found in ~8% of titles)")
    print("   - Remove series titles embedded in book titles (67.5% of series books)")
    print("   - Standardize numbering formats (Book 1, Volume 2, etc.)")
    
    print("\n2. AUTHOR NAME NORMALIZATION:")
    print("   - Resolve 14,634 potential duplicate author names")
    print("   - Standardize name formats (mean: 2.1 words, median: 2.0 words)")
    print("   - Handle pen names and variations")
    
    print("\n3. DESCRIPTION TEXT CLEANING:")
    print("   - Remove HTML tags (13 descriptions affected)")
    print("   - Clean HTML entities (55 descriptions affected)")
    print("   - Normalize whitespace (94.8% have multiple whitespace)")
    print("   - Handle line breaks/tabs (83.7% affected)")
    print("   - Handle missing descriptions (5.2% missing)")
    
    print("\n4. SERIES HANDLING:")
    print("   - 66.9% of books are in series")
    print("   - Most common series sizes: 2-5 books")
    print("   - Extract series information consistently")
    print("   - Create clean series titles")
    
    print("\n5. SUBGENRE CLASSIFICATION:")
    print("   - Parse popular shelves for subgenre signals")
    print("   - Create standardized subgenre categories")
    print("   - Handle overlapping subgenres")
    
    print("\n6. DATA QUALITY IMPROVEMENTS:")
    print("   - Fill missing median_publication_year (93% missing)")
    print("   - Handle missing num_pages_median (30% missing)")
    print("   - Resolve missing series information (33% missing)")
    
    print("\n✅ EDA Analysis Part 3 completed successfully!")
    print("📊 Comprehensive analysis complete. Ready for cleaning implementation.")

if __name__ == "__main__":
    main()
