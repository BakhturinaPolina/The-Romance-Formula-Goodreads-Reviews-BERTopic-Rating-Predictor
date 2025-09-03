#!/usr/bin/env python3
"""
Dataset Structure Analysis for Step 2 Planning
Analyze the cleaned dataset structure to understand duplicates and inconsistencies.
"""

import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from data_quality.data_quality_assessment import DataQualityAssessment


def analyze_dataset_structure():
    """Analyze the cleaned dataset structure for Step 2 planning."""
    print("🔍 Dataset Structure Analysis for Step 2 Planning")
    print("=" * 60)
    print("Analyzing cleaned dataset to identify duplicate detection strategies")
    print("=" * 60)
    
    # Initialize assessor and load the cleaned dataset
    assessor = DataQualityAssessment()
    
    # Look for the cleaned dataset specifically
    cleaned_files = list(Path("../../data/processed").glob("final_books_*_cleaned_nlp_ready_*.csv"))
    if not cleaned_files:
        print("❌ No cleaned dataset found. Please run create_cleaned_dataset.py first.")
        return
    
    # Use the most recent cleaned dataset
    cleaned_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    cleaned_file = cleaned_files[0]
    print(f"📚 Loading cleaned dataset: {cleaned_file.name}")
    
    try:
        data = pd.read_csv(cleaned_file)
        print(f"✅ Dataset loaded: {len(data):,} books, {len(data.columns)} columns")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"  - Total books: {len(data):,}")
    print(f"  - Total columns: {len(data.columns)}")
    print(f"  - Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Analyze column structure
    print(f"\n📋 COLUMN ANALYSIS:")
    for i, col in enumerate(data.columns, 1):
        dtype = str(data[col].dtype)
        null_count = data[col].isnull().sum()
        null_pct = (null_count / len(data)) * 100
        unique_count = data[col].nunique()
        
        print(f"  {i:2d}. {col:25} | {dtype:10} | {null_count:6,} null ({null_pct:5.1f}%) | {unique_count:6,} unique")
    
    # Analyze key fields for duplicate detection
    print(f"\n🔍 DUPLICATE DETECTION ANALYSIS:")
    
    # 1. Work ID analysis
    work_id_duplicates = data['work_id'].duplicated().sum()
    work_id_unique = data['work_id'].nunique()
    print(f"  📚 Work ID:")
    print(f"    - Unique work IDs: {work_id_unique:,}")
    print(f"    - Duplicate work IDs: {work_id_duplicates:,}")
    print(f"    - Expected: {len(data):,} (should match total books)")
    
    if work_id_duplicates > 0:
        print(f"    ⚠️  WARNING: Found {work_id_duplicates} duplicate work IDs!")
        # Show examples of duplicates
        duplicate_work_ids = data[data['work_id'].duplicated(keep=False)]['work_id'].value_counts().head(5)
        print(f"    - Top duplicate work IDs: {duplicate_work_ids.to_dict()}")
    
    # 2. Title analysis
    title_duplicates = data['title'].duplicated().sum()
    title_unique = data['title'].nunique()
    print(f"  📖 Title:")
    print(f"    - Unique titles: {title_unique:,}")
    print(f"    - Duplicate titles: {title_duplicates:,}")
    print(f"    - Duplication rate: {title_duplicates/len(data)*100:.1f}%")
    
    if title_duplicates > 0:
        print(f"    ⚠️  Found {title_duplicates} duplicate titles")
        # Show examples of duplicate titles
        duplicate_titles = data[data['title'].duplicated(keep=False)]['title'].value_counts().head(5)
        print(f"    - Top duplicate titles: {duplicate_titles.to_dict()}")
    
    # 3. Author analysis
    author_id_duplicates = data['author_id'].duplicated().sum()
    author_id_unique = data['author_id'].nunique()
    print(f"  👤 Author ID:")
    print(f"    - Unique authors: {author_id_unique:,}")
    print(f"    - Duplicate author IDs: {author_id_duplicates:,}")
    print(f"    - Books per author (avg): {len(data)/author_id_unique:.1f}")
    
    # 4. Series analysis
    series_id_analysis = data['series_id'].value_counts()
    series_count = len(series_id_analysis)
    series_with_multiple = (series_id_analysis > 1).sum()
    print(f"  📚 Series:")
    print(f"    - Total series: {series_count:,}")
    print(f"    - Series with multiple books: {series_with_multiple:,}")
    print(f"    - Books in series: {series_id_analysis.sum():,}")
    print(f"    - Standalone books: {len(data) - series_id_analysis.sum():,}")
    
    # 5. Genre analysis
    genre_analysis = data['genres'].value_counts()
    top_genres = genre_analysis.head(10)
    print(f"  🏷️  Genres:")
    print(f"    - Unique genre combinations: {len(genre_analysis):,}")
    print(f"    - Top 10 genre combinations:")
    for i, (genre, count) in enumerate(top_genres.items(), 1):
        print(f"      {i:2d}. {genre:50} | {count:5,} books")
    
    # Analyze potential inconsistencies
    print(f"\n⚠️  POTENTIAL INCONSISTENCIES TO INVESTIGATE:")
    
    # 1. Series ordering inconsistencies
    if 'series_works_count' in data.columns:
        series_inconsistencies = []
        for series_id in data['series_id'].dropna().unique():
            series_books = data[data['series_id'] == series_id]
            expected_count = series_books['series_works_count'].iloc[0]
            actual_count = len(series_books)
            if pd.notna(expected_count) and expected_count != actual_count:
                series_inconsistencies.append({
                    'series_id': series_id,
                    'expected': expected_count,
                    'actual': actual_count,
                    'difference': actual_count - expected_count
                })
        
        if series_inconsistencies:
            print(f"  📚 Series count inconsistencies: {len(series_inconsistencies)} found")
            for inc in series_inconsistencies[:5]:  # Show first 5
                print(f"    - Series {inc['series_id']:.0f}: expected {inc['expected']}, found {inc['actual']} (diff: {inc['difference']:+.0f})")
        else:
            print(f"  ✅ Series counts are consistent")
    
    # 2. Publication year anomalies
    year_stats = data['publication_year'].describe()
    print(f"  📅 Publication year range: {year_stats['min']:.0f} - {year_stats['max']:.0f}")
    print(f"    - Mean: {year_stats['mean']:.1f}")
    print(f"    - Median: {year_stats['50%']:.0f}")
    
    # Check for unusual years
    unusual_years = data[~data['publication_year'].between(2000, 2020)]
    if len(unusual_years) > 0:
        print(f"    ⚠️  Found {len(unusual_years)} books outside 2000-2020 range")
        print(f"    - Years: {sorted(unusual_years['publication_year'].unique())}")
    
    # 3. Rating anomalies
    rating_stats = data['average_rating_weighted_mean'].describe()
    print(f"  ⭐ Rating statistics:")
    print(f"    - Range: {rating_stats['min']:.2f} - {rating_stats['max']:.2f}")
    print(f"    - Mean: {rating_stats['mean']:.2f}")
    print(f"    - Median: {rating_stats['50%']:.2f}")
    
    # Check for extreme ratings
    extreme_ratings = data[~data['average_rating_weighted_mean'].between(0, 5)]
    if len(extreme_ratings) > 0:
        print(f"    ⚠️  Found {len(extreme_ratings)} books with ratings outside 0-5 range")
    
    # 4. Page count anomalies
    page_stats = data['num_pages_median'].describe()
    print(f"  📏 Page count statistics:")
    print(f"    - Range: {page_stats['min']:.0f} - {page_stats['max']:.0f}")
    print(f"    - Mean: {page_stats['mean']:.1f}")
    print(f"    - Median: {page_stats['50%']:.0f}")
    
    # Check for extreme page counts
    extreme_pages = data[~data['num_pages_median'].between(10, 2000)]
    if len(extreme_pages) > 0:
        print(f"    ⚠️  Found {len(extreme_pages)} books with extreme page counts")
        print(f"    - Counts: {sorted(extreme_pages['num_pages_median'].unique())}")
    
    # Summary and recommendations
    print(f"\n📋 STEP 2 IMPLEMENTATION PLAN:")
    print(f"  Based on this analysis, here's what to focus on:")
    
    if work_id_duplicates > 0:
        print(f"  1. 🔴 CRITICAL: Fix {work_id_duplicates} duplicate work IDs")
    
    if title_duplicates > 0:
        print(f"  2. 🟡 HIGH: Investigate {title_duplicates} duplicate titles")
    
    if series_inconsistencies:
        print(f"  3. 🟡 HIGH: Fix {len(series_inconsistencies)} series count inconsistencies")
    
    print(f"  4. 🟢 MEDIUM: Validate publication year ranges")
    print(f"  5. 🟢 MEDIUM: Check rating and page count distributions")
    print(f"  6. 🟢 MEDIUM: Analyze genre classification patterns")
    
    return data


if __name__ == "__main__":
    data = analyze_dataset_structure()
    
    if data is not None:
        print(f"\n✅ Dataset structure analysis completed!")
        print(f"📝 Use this analysis to plan Step 2 implementation.")
    else:
        print(f"\n❌ Analysis failed!")
        sys.exit(1)
