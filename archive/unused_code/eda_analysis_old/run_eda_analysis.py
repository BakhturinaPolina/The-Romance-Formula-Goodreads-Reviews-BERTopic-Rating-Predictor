#!/usr/bin/env python3
"""
Run EDA Analysis for Final Dataset
Executes the analysis step by step to examine data quality and suggest cleaning steps.
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
    """Run the EDA analysis step by step."""
    print("🔍 STARTING EDA ANALYSIS OF FINAL DATASET")
    print("=" * 60)
    
    # Step 1: Import libraries
    print("✅ Libraries imported successfully")
    
    # Step 2: Load dataset
    print("\n📚 Loading dataset...")
    dataset_path = "data/processed/final_books_2000_2020_en_20250901_024106.csv"
    
    try:
        df = pd.read_csv(dataset_path)
        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Step 3: Dataset Overview
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
    
    # Step 4: Missing Values Analysis
    print("\n🔍 MISSING VALUES ANALYSIS")
    print("=" * 50)
    
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing_Count': missing_data.values,
        'Missing_Percent': missing_percent.values
    })
    
    missing_df = missing_df.sort_values('Missing_Percent', ascending=False)
    print(missing_df)
    
    # Step 5: Title Analysis
    print("\n🔍 TITLE ANALYSIS")
    print("=" * 50)
    
    # Basic title statistics
    df['title_length'] = df['title'].str.len()
    df['title_word_count'] = df['title'].str.split().str.len()
    
    print(f"📖 Title length statistics:")
    print(f"   - Mean length: {df['title_length'].mean():.1f} characters")
    print(f"   - Median length: {df['title_length'].median():.1f} characters")
    print(f"   - Min length: {df['title_length'].min()} characters")
    print(f"   - Max length: {df['title_length'].max()} characters")
    
    print(f"\n📝 Title word count statistics:")
    print(f"   - Mean words: {df['title_word_count'].mean():.1f} words")
    print(f"   - Median words: {df['title_word_count'].median():.1f} words")
    print(f"   - Min words: {df['title_word_count'].min()} words")
    print(f"   - Max words: {df['title_word_count'].max()} words")
    
    # Step 6: Series Patterns in Titles
    print("\n🔍 SERIES PATTERNS IN TITLES")
    print("=" * 50)
    
    # Common series indicators
    series_patterns = [
        r'\b(\d+)\s*[:\-]\s*',  # Number followed by : or -
        r'\b(Book|Volume|Part)\s+(\d+)\b',  # Book 1, Volume 2, etc.
        r'\b(\d+)\s*(?:st|nd|rd|th)\s*',  # 1st, 2nd, 3rd, etc.
        r'\b(\d+)\s*$',  # Number at end
        r'\b(\d+)\s*\('  # Number followed by parenthesis
    ]
    
    pattern_names = ['Number:Colon', 'Book/Volume/Part', 'Ordinal', 'End Number', 'Number(']
    
    for pattern, name in zip(series_patterns, pattern_names):
        matches = df['title'].str.contains(pattern, regex=True, na=False)
        count = matches.sum()
        percentage = (count / len(df)) * 100
        print(f"{name}: {count:,} titles ({percentage:.1f}%)")
    
    # Show examples of titles with series patterns
    print("\n📚 Examples of titles with series patterns:")
    for pattern, name in zip(series_patterns, pattern_names):
        matches = df[df['title'].str.contains(pattern, regex=True, na=False)]
        if not matches.empty:
            print(f"\n{name} examples:")
            for title in matches['title'].head(3):
                print(f"  - {title}")
    
    print("\n✅ EDA Analysis completed successfully!")
    print("📊 Check the output above for data quality insights and cleaning recommendations.")

if __name__ == "__main__":
    main()
