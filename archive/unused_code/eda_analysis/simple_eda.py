#!/usr/bin/env python3
"""
Simple EDA Analysis for Romance Novel Dataset
Basic exploratory data analysis without complex features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def run_simple_eda(df: pd.DataFrame, dataset_name: str = "Dataset"):
    """
    Run simple EDA analysis on the dataset.
    
    Args:
        df: DataFrame to analyze
        dataset_name: Name of the dataset for display
    """
    print(f"🔍 SIMPLE EDA ANALYSIS: {dataset_name}")
    print("=" * 60)
    
    # Basic overview
    print(f"📊 BASIC OVERVIEW:")
    print(f"  - Shape: {df.shape}")
    print(f"  - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"  - Columns: {list(df.columns)}")
    
    # Data types
    print(f"\n📋 DATA TYPES:")
    print(df.dtypes)
    
    # Missing values
    print(f"\n❌ MISSING VALUES:")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing_Count': missing_data.values,
        'Missing_Percent': missing_percent.values
    }).sort_values('Missing_Count', ascending=False)
    
    print(missing_df)
    
    # Publication year analysis
    if 'publication_year' in df.columns:
        print(f"\n📅 PUBLICATION YEAR ANALYSIS:")
        print(f"  - Range: {df['publication_year'].min()} - {df['publication_year'].max()}")
        print(f"  - Mean: {df['publication_year'].mean():.1f}")
        print(f"  - Median: {df['publication_year'].median():.1f}")
        
        # Year distribution
        year_counts = df['publication_year'].value_counts().sort_index()
        print(f"  - Most common year: {year_counts.idxmax()} ({year_counts.max():,} books)")
    
    # Title analysis
    if 'title' in df.columns:
        print(f"\n📖 TITLE ANALYSIS:")
        df['title_length'] = df['title'].str.len()
        df['title_word_count'] = df['title'].str.split().str.len()
        
        print(f"  - Mean length: {df['title_length'].mean():.1f} characters")
        print(f"  - Median length: {df['title_length'].median():.1f} characters")
        print(f"  - Mean word count: {df['title_word_count'].mean():.1f} words")
        print(f"  - Median word count: {df['title_word_count'].median():.1f} words")
        
        # Show some sample titles
        print(f"\n📚 SAMPLE TITLES:")
        for i, title in enumerate(df['title'].head(5)):
            print(f"  {i+1}. {title}")
    
    # Author analysis
    if 'author_name' in df.columns:
        print(f"\n👤 AUTHOR ANALYSIS:")
        print(f"  - Unique authors: {df['author_name'].nunique():,}")
        print(f"  - Books per author (mean): {len(df) / df['author_name'].nunique():.1f}")
        
        # Top authors
        top_authors = df['author_name'].value_counts().head(5)
        print(f"\n🏆 TOP 5 AUTHORS:")
        for author, count in top_authors.items():
            print(f"  - {author}: {count} books")
    
    # Series analysis
    if 'series_id' in df.columns:
        print(f"\n📚 SERIES ANALYSIS:")
        series_coverage = df['series_id'].notna().sum()
        print(f"  - Books in series: {series_coverage:,} ({series_coverage/len(df)*100:.1f}%)")
        print(f"  - Books not in series: {(~df['series_id'].notna()).sum():,} ({(~df['series_id'].notna()).sum()/len(df)*100:.1f}%)")
        
        if 'series_title' in df.columns:
            series_titles = df['series_title'].value_counts().head(5)
            print(f"\n📖 TOP 5 SERIES:")
            for series, count in series_titles.head(5).items():
                if pd.notna(series):
                    print(f"  - {series}: {count} books")
    
    # Ratings analysis
    if 'average_rating_weighted_mean' in df.columns:
        print(f"\n⭐ RATINGS ANALYSIS:")
        print(f"  - Average rating (mean): {df['average_rating_weighted_mean'].mean():.2f}")
        print(f"  - Average rating (median): {df['average_rating_weighted_mean'].median():.2f}")
        print(f"  - Min rating: {df['average_rating_weighted_mean'].min():.2f}")
        print(f"  - Max rating: {df['average_rating_weighted_mean'].max():.2f}")
    
    # Title cleaning results (if available)
    if 'title_cleaned' in df.columns and 'title_original' in df.columns:
        print(f"\n🧹 TITLE CLEANING RESULTS:")
        titles_cleaned = (df['title_original'] != df['title_cleaned']).sum()
        print(f"  - Titles cleaned: {titles_cleaned:,} ({titles_cleaned/len(df)*100:.1f}%)")
        print(f"  - Titles unchanged: {len(df) - titles_cleaned:,} ({(len(df) - titles_cleaned)/len(df)*100:.1f}%)")
        
        # Show examples of cleaned titles
        if titles_cleaned > 0:
            print(f"\n🔍 EXAMPLES OF CLEANED TITLES:")
            cleaned_examples = df[df['title_original'] != df['title_cleaned']].head(3)
            for _, row in cleaned_examples.iterrows():
                print(f"  - '{row['title_original']}' → '{row['title_cleaned']}'")
    
    print(f"\n✅ Simple EDA analysis completed!")

def main():
    """Run simple EDA on the basic cleaned dataset."""
    print("🚀 Running Simple EDA Analysis")
    print("=" * 50)
    
    # Try to find the basic cleaned dataset
    processed_dir = Path("data/processed")
    basic_cleaned_files = list(processed_dir.glob("romance_novels_basic_cleaned_*.csv"))
    
    if basic_cleaned_files:
        # Use the most recent basic cleaned file
        input_path = sorted(basic_cleaned_files)[-1]
        print(f"📚 Found basic cleaned dataset: {input_path.name}")
    else:
        # Fallback to the 500-book sample
        input_path = "data/processed/final_books_2000_2020_en_cleaned_titles_sampled_500_20250901_221322.csv"
        print(f"📚 Using fallback dataset: {input_path}")
    
    if not Path(input_path).exists():
        print(f"❌ Dataset not found: {input_path}")
        print("Please run the basic cleaner first to generate the cleaned dataset.")
        return
    
    # Load and analyze dataset
    print(f"📖 Loading dataset...")
    df = pd.read_csv(input_path)
    print(f"✅ Dataset loaded successfully!")
    
    # Run simple EDA
    run_simple_eda(df, input_path.name)
    
    print(f"\n🎉 Simple EDA completed successfully!")

if __name__ == "__main__":
    main()
