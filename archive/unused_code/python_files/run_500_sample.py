#!/usr/bin/env python3
"""
Simple script to run CSV builder with 500 book sample.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from csv_building.final_csv_builder_working import OptimizedFinalCSVBuilder

def main():
    """Run the CSV builder with 500 book sample."""
    print("🚀 Running CSV Builder with 500 Book Sample")
    print("=" * 50)
    
    # Initialize the builder
    builder = OptimizedFinalCSVBuilder()
    
    print("📊 Processing sample of 500 works...")
    
    # Run with 500 book sample
    output_path = builder.build_final_csv_optimized(sample_size=500)
    
    print(f"\n✅ CSV with 500 books generated: {output_path}")
    
    # Load and show summary
    import pandas as pd
    df = pd.read_csv(output_path)
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📅 Publication year range: {df['publication_year'].min()} - {df['publication_year'].max()}")
    print(f"📚 Total works: {len(df):,}")
    print(f"👤 Works with author data: {df['author_id'].notna().sum():,}")
    print(f"📖 Works with series data: {df['series_id'].notna().sum():,}")
    
    # Show title cleaning metrics
    if 'title_cleaned' in df.columns:
        titles_cleaned = df['title_cleaned'].sum()
        print(f"\n📖 Title Cleaning Results:")
        print(f"  - Titles cleaned: {titles_cleaned:,} ({titles_cleaned/len(df)*100:.1f}%)")
        print(f"  - Titles unchanged: {len(df) - titles_cleaned:,} ({(len(df) - titles_cleaned)/len(df)*100:.1f}%)")
    
    return output_path

if __name__ == "__main__":
    main()
