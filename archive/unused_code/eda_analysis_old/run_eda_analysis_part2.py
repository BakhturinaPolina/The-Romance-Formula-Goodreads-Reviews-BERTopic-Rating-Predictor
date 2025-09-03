#!/usr/bin/env python3
"""
Run EDA Analysis Part 2: Author, Description, and Series Analysis
Continues the analysis from the first part to examine more aspects of the dataset.
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
    """Run the second part of EDA analysis."""
    print("🔍 CONTINUING EDA ANALYSIS - PART 2")
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
    
    # Add derived columns from previous analysis
    df['title_length'] = df['title'].str.len()
    df['title_word_count'] = df['title'].str.split().str.len()
    
    # Step 7: Author Name Analysis
    print("\n🔍 AUTHOR NAME ANALYSIS")
    print("=" * 50)
    
    # Basic author statistics
    print(f"👤 Total unique authors: {df['author_id'].nunique():,}")
    print(f"📚 Books per author (mean): {len(df) / df['author_id'].nunique():.1f}")
    print(f"📚 Books per author (median): {df.groupby('author_id').size().median():.1f}")
    
    # Author name length analysis
    df['author_name_length'] = df['author_name'].str.len()
    df['author_name_word_count'] = df['author_name'].str.split().str.len()
    
    print(f"\n📝 Author name statistics:")
    print(f"   - Mean name length: {df['author_name_length'].mean():.1f} characters")
    print(f"   - Median name length: {df['author_name_length'].median():.1f} characters")
    print(f"   - Mean word count: {df['author_name_word_count'].mean():.1f} words")
    print(f"   - Median word count: {df['author_name_word_count'].median():.1f} words")
    
    # Step 8: Author Name Variations Analysis
    print("\n🔍 AUTHOR NAME VARIATIONS ANALYSIS")
    print("=" * 50)
    
    # Check for authors with multiple name variations
    author_name_counts = df.groupby('author_id')['author_name'].nunique()
    multiple_names = author_name_counts[author_name_counts > 1]
    
    print(f"👤 Authors with multiple name variations: {len(multiple_names):,}")
    if not multiple_names.empty:
        print(f"\n📚 Examples of authors with multiple names:")
        for author_id in multiple_names.head(5).index:
            names = df[df['author_id'] == author_id]['author_name'].unique()
            print(f"  Author ID {author_id}: {names}")
    
    # Check for potential duplicate authors (same name, different ID)
    author_name_to_ids = defaultdict(list)
    for _, row in df.iterrows():
        author_name_to_ids[row['author_name']].append(row['author_id'])
    
    duplicate_names = {name: ids for name, ids in author_name_to_ids.items() if len(ids) > 1}
    print(f"\n⚠️  Potential duplicate author names: {len(duplicate_names):,}")
    
    if duplicate_names:
        print(f"\n📚 Examples of potential duplicate names:")
        for name, ids in list(duplicate_names.items())[:5]:
            print(f"  '{name}': {ids}")
    
    # Step 9: Description Text Analysis
    print("\n🔍 DESCRIPTION TEXT ANALYSIS")
    print("=" * 50)
    
    # Basic description statistics
    df['description_length'] = df['description'].str.len()
    df['description_word_count'] = df['description'].str.split().str.len()
    
    print(f"📖 Description statistics:")
    print(f"   - Mean length: {df['description_length'].mean():.1f} characters")
    print(f"   - Median length: {df['description_length'].median():.1f} characters")
    print(f"   - Min length: {df['description_length'].min()} characters")
    print(f"   - Max length: {df['description_length'].max()} characters")
    print(f"   - Mean words: {df['description_word_count'].mean():.1f} words")
    print(f"   - Median words: {df['description_word_count'].median():.1f} words")
    
    # Check for missing descriptions
    missing_descriptions = df['description'].isnull().sum()
    print(f"\n❌ Missing descriptions: {missing_descriptions:,} ({missing_descriptions/len(df)*100:.1f}%)\n")
    
    # Check for very short descriptions (potential data quality issues)
    short_descriptions = (df['description_length'] < 50).sum()
    print(f"📝 Very short descriptions (<50 chars): {short_descriptions:,} ({short_descriptions/len(df)*100:.1f}%)")
    
    # Step 10: HTML and Special Characters Analysis
    print("\n🔍 HTML AND SPECIAL CHARACTERS IN DESCRIPTIONS")
    print("=" * 50)
    
    # Common HTML patterns
    html_patterns = [
        r'<[^>]+>',  # HTML tags
        r'&[a-zA-Z]+;',  # HTML entities
        r'\s+',  # Multiple whitespace
        r'[\r\n\t]+',  # Line breaks and tabs
        r'[\u00A0-\uFFFF]',  # Non-ASCII characters
    ]
    
    pattern_names = ['HTML Tags', 'HTML Entities', 'Multiple Whitespace', 'Line Breaks/Tabs', 'Non-ASCII']
    
    for pattern, name in zip(html_patterns, pattern_names):
        matches = df['description'].str.contains(pattern, regex=True, na=False)
        count = matches.sum()
        percentage = (count / len(df)) * 100
        print(f"{name}: {count:,} descriptions ({percentage:.1f}%)")
    
    # Show examples of descriptions with HTML
    html_descriptions = df[df['description'].str.contains(r'<[^>]+>', regex=True, na=False)]
    if not html_descriptions.empty:
        print(f"\n📚 Examples of descriptions with HTML:")
        for desc in html_descriptions['description'].head(3):
            print(f"  - {desc[:200]}...")
    
    # Step 11: Series Pattern Analysis
    print("\n🔍 SERIES PATTERN ANALYSIS")
    print("=" * 50)
    
    # Series coverage
    series_coverage = df['series_id'].notna().sum()
    print(f"📚 Books in series: {series_coverage:,} ({series_coverage/len(df)*100:.1f}%)")
    print(f"📚 Books not in series: {(~df['series_id'].notna()).sum():,} ({(~df['series_id'].notna()).sum()/len(df)*100:.1f}%)")
    
    # Series size distribution
    series_sizes = df.groupby('series_id')['series_works_count'].first().value_counts().sort_index()
    print(f"\n📊 Series size distribution:")
    for size, count in series_sizes.head(10).items():
        print(f"  {size} books: {count:,} series")
    
    # Check relationship between series titles and book titles
    print(f"\n🔍 SERIES TITLE VS BOOK TITLE RELATIONSHIP")
    series_books = df[df['series_id'].notna()].copy()
    series_books['title_contains_series'] = series_books.apply(
        lambda row: row['series_title'].lower() in row['title'].lower() if pd.notna(row['series_title']) else False, 
        axis=1
    )
    
    contains_series = series_books['title_contains_series'].sum()
    print(f"📚 Books with series titles embedded in book titles: {contains_series:,} ({contains_series/len(series_books)*100:.1f}%)")
    
    # Show examples
    if not series_books.empty:
        examples = series_books[series_books['title_contains_series']].head(5)
        print(f"\n📚 Examples of books with embedded series titles:")
        for _, row in examples.iterrows():
            print(f"  Series: '{row['series_title']}' | Book: '{row['title']}'")
    
    print("\n✅ EDA Analysis Part 2 completed successfully!")
    print("📊 Check the output above for detailed insights on authors, descriptions, and series.")

if __name__ == "__main__":
    main()
