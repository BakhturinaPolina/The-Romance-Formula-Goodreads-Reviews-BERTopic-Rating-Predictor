#!/usr/bin/env python
# coding: utf-8

"""
02: Final Dataset EDA and Cleaning Recommendations

Objective: Comprehensive exploration of the final processed romance novel dataset 
to identify cleaning opportunities and prepare for NLP analysis.

Research Context: Analyze how thematic characteristics of modern romance novels 
relate to reader engagement/popularity using Goodreads metadata.

Dataset: final_books_2000_2020_en_20250901_024106.csv (119,678 romance novels)

Analysis Plan:
1. Dataset Overview - Basic structure, data types, missing values
2. Title Analysis - Series patterns, numbering, cleaning opportunities
3. Author Name Analysis - Duplicates, variations, normalization needs
4. Description Text Analysis - Text quality, HTML artifacts, length distributions
5. Series Pattern Analysis - Series titles and book title relationships
6. Publication & Popularity Analysis - Temporal trends and engagement metrics
7. Subgenre Signal Analysis - Popular shelves and genre classification
8. Cleaning Recommendations - Specific suggestions with code examples

Expected Outputs:
- Data quality assessment
- Title and series cleaning patterns
- Author name normalization strategies
- Text preprocessing recommendations
- Final dataset preparation for NLP analysis
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
import os

# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 1000)

def main():
    """Main EDA analysis function"""
    print("🚀 Starting Final Dataset EDA and Cleaning Analysis")
    print("=" * 60)
    
    # Import required libraries
    print("📚 Importing libraries...")
    print("✅ Libraries imported successfully")
    
    # Load the final processed dataset - use absolute path from project root
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    dataset_path = project_root / "data" / "processed" / "final_books_2000_2020_en_20250901_024106.csv"
    
    print(f"📚 Loading dataset from: {dataset_path}")
    
    try:
        # Load with progress indicator
        df = pd.read_csv(dataset_path)
        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
    except FileNotFoundError:
        print(f"❌ Error: Dataset not found at {dataset_path}")
        print("Please ensure the dataset exists and the path is correct.")
        return
    
    # Run all analysis sections
    run_dataset_overview(df)
    run_missing_values_analysis(df)
    run_title_analysis(df)
    run_author_analysis(df)
    run_description_analysis(df)
    run_series_analysis(df)
    run_publication_analysis(df)
    run_cleaning_functions(df)
    run_cleaning_tests(df)
    
    print("\n🎉 EDA Analysis Complete!")
    print("=" * 60)

def run_dataset_overview(df):
    """Run dataset overview analysis"""
    print("\n🔍 DATASET OVERVIEW")
    print("=" * 50)
    
    # Basic info
    print(f"📚 Total records: {len(df):,}")
    print(f"📋 Total columns: {len(df.columns)}")
    print(f"📅 Publication year range: {df['publication_year'].min()} - {df['publication_year'].max()}")
    
    # Data types
    print("\n📊 Data Types:")
    print(df.dtypes)
    
    # Memory usage
    memory_usage = df.memory_usage(deep=True).sum() / 1024**2
    print(f"\n💾 Memory usage: {memory_usage:.2f} MB")

def run_missing_values_analysis(df):
    """Run missing values analysis"""
    print("\n🔍 MISSING VALUES ANALYSIS")
    print("=" * 50)
    
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing_Count': missing_data.values,
        'Missing_Percent': missing_percent.values
    }).sort_values('Missing_Count', ascending=False)
    
    print("📊 Missing values summary:")
    print(missing_df)
    
    # Check for columns with high missing values
    high_missing = missing_df[missing_df['Missing_Percent'] > 50]
    if not high_missing.empty:
        print(f"\n⚠️  Columns with >50% missing values: {len(high_missing)}")
        for _, row in high_missing.iterrows():
            print(f"  - {row['Column']}: {row['Missing_Percent']:.1f}%")

def run_title_analysis(df):
    """Run title analysis"""
    print("\n🔍 TITLE ANALYSIS")
    print("=" * 50)
    
    # Basic title statistics
    df['title_length'] = df['title'].str.len()
    df['title_word_count'] = df['title'].str.split().str.len()
    
    print(f"📝 Title statistics:")
    print(f"   - Mean length: {df['title_length'].mean():.1f} characters")
    print(f"   - Median length: {df['title_length'].median():.1f} characters")
    print(f"   - Mean word count: {df['title_word_count'].mean():.1f} words")
    print(f"   - Median word count: {df['title_word_count'].median():.1f} words")
    
    # Check for titles with series information
    series_patterns = [
        r'\b\d+\s*[:\-]\s*',  # Number followed by : or -
        r'\b(Book|Volume|Part)\s+\d+\b',  # Book 1, Volume 2, etc.
        r'\b\d+\s*(?:st|nd|rd|th)\s*',  # 1st, 2nd, 3rd, etc.
        r'\b\d+\s*\('  # Number followed by parenthesis
    ]
    
    titles_with_series = 0
    for pattern in series_patterns:
        matches = df['title'].str.contains(pattern, regex=True, na=False)
        titles_with_series += matches.sum()
    
    print(f"\n📚 Titles with series patterns: {titles_with_series:,} ({titles_with_series/len(df)*100:.1f}%)")

def run_author_analysis(df):
    """Run author analysis"""
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
    
    # Check for potential author name duplicates/variations
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

def run_description_analysis(df):
    """Run description analysis"""
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
    print(f"\n❌ Missing descriptions: {missing_descriptions:,} ({missing_descriptions/len(df)*100:.1f}%)")
    
    # Check for very short descriptions (potential data quality issues)
    short_descriptions = (df['description_length'] < 50).sum()
    print(f"📝 Very short descriptions (<50 chars): {short_descriptions:,} ({short_descriptions/len(df)*100:.1f}%)")
    
    # Check for HTML artifacts and special characters in descriptions
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

def run_series_analysis(df):
    """Run series analysis"""
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

def run_publication_analysis(df):
    """Run publication and popularity analysis"""
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

def run_cleaning_functions(df):
    """Define and demonstrate cleaning functions"""
    print("\n🧹 CLEANING FUNCTIONS")
    print("=" * 50)
    
    def clean_title(title, series_title=None):
        """Clean book title by removing series information."""
        if pd.isna(title):
            return title
        
        # Remove series title if embedded
        if series_title and pd.notna(series_title):
            title = title.replace(series_title, '').strip()
        
        # Remove common series patterns
        patterns = [
            r'\b\d+\s*[:\-]\s*',  # Number followed by : or -
            r'\b(Book|Volume|Part)\s+\d+\b',  # Book 1, Volume 2, etc.
            r'\b\d+\s*(?:st|nd|rd|th)\s*',  # 1st, 2nd, 3rd, etc.
            r'\b\d+\s*\('  # Number followed by parenthesis
        ]
        
        for pattern in patterns:
            title = re.sub(pattern, '', title)
        
        # Clean up whitespace
        title = re.sub(r'\s+', ' ', title).strip()
        
        return title if title else "Untitled"
    
    def clean_description(description):
        """Clean book description by removing HTML and normalizing text."""
        if pd.isna(description):
            return description
        
        # Remove HTML tags
        cleaned = re.sub(r'<[^>]+>', '', description)
        # Remove HTML entities
        cleaned = re.sub(r'&[a-zA-Z]+;', ' ', cleaned)
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        # Remove line breaks and tabs
        cleaned = re.sub(r'[\r\n\t]+', ' ', cleaned)
        # Clean up
        cleaned = cleaned.strip()
        
        return cleaned if cleaned else description
    
    def extract_series_number(title):
        """Extract series number from title."""
        if pd.isna(title):
            return None
        
        # Common patterns
        patterns = [
            r'\b(\d+)\s*[:\-]\s*',  # Number followed by : or -
            r'\b(Book|Volume|Part)\s+(\d+)\b',  # Book 1, Volume 2, etc.
            r'\b(\d+)\s*(?:st|nd|rd|th)\s*',  # 1st, 2nd, 3rd, etc.
            r'\b(\d+)\s*\('  # Number followed by parenthesis
        ]
        
        for pattern in patterns:
            match = re.search(pattern, title)
            if match:
                return int(match.group(1) if len(match.groups()) > 1 else match.group(1))
        
        return None
    
    print("✅ Sample cleaning functions defined:")
    print("   - clean_title(): Remove series information from titles")
    print("   - clean_description(): Remove HTML and normalize text")
    print("   - extract_series_number(): Extract series numbers from titles")
    
    # Store functions for testing
    global clean_title_func, clean_description_func, extract_series_number_func
    clean_title_func = clean_title
    clean_description_func = clean_description
    extract_series_number_func = extract_series_number

def run_cleaning_tests(df):
    """Test cleaning functions on sample data"""
    print("\n🧪 TESTING CLEANING FUNCTIONS")
    print("=" * 50)
    
    # Test on sample data
    sample_data = df[['title', 'series_title', 'description']].head(5)
    print("📚 Sample data before cleaning:")
    print(sample_data)
    
    print("\n🧹 After cleaning:")
    for idx, row in sample_data.iterrows():
        print(f"\nBook {idx}:")
        print(f"  Original title: {row['title']}")
        print(f"  Cleaned title: {clean_title_func(row['title'], row['series_title'])}")
        print(f"  Series number: {extract_series_number_func(row['title'])}")
        if pd.notna(row['description']):
            desc_preview = row['description'][:100] + "..." if len(row['description']) > 100 else row['description']
            cleaned_desc = clean_description_func(row['description'])
            cleaned_preview = cleaned_desc[:100] + "..." if len(cleaned_desc) > 100 else cleaned_desc
            print(f"  Description preview: {desc_preview}")
            print(f"  Cleaned description: {cleaned_preview}")

if __name__ == "__main__":
    main()
