"""
Run the FAST cleaned titles CSV builder with performance optimizations.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from csv_building.final_csv_builder_cleaned_titles_fast import FastFinalCSVBuilderCleanedTitles


def main():
    """Run the fast cleaned titles CSV builder."""
    print("🚀 Running FAST Final CSV Builder with Cleaned Titles...")
    print("📖 Uses ONLY works.original_title with simple bracket stripping")
    print("⚡ Performance optimized for speed!")
    
    # Initialize the fast cleaned titles builder
    builder = FastFinalCSVBuilderCleanedTitles()
    
    # Ask user for sample size or full dataset
    print("\n📊 Choose processing mode:")
    print("1. Test with sample (recommended for first run)")
    print("2. Process full dataset")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        try:
            sample_size = int(input("Enter sample size (e.g., 100): ").strip())
            print(f"\n📊 Processing sample of {sample_size} works with FAST processing...")
            output_path = builder.build_final_csv_fast(sample_size=sample_size)
        except ValueError:
            print("Invalid sample size, using default of 100")
            output_path = builder.build_final_csv_fast(sample_size=100)
    else:
        print("\n📊 Processing full dataset with FAST processing...")
        output_path = builder.build_final_csv_fast()  # No sample size = full dataset
    
    print(f"\n✅ Fast cleaned titles CSV generated: {output_path}")
    
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
    
    # Show sample of cleaned titles
    if 'title_cleaned' in df.columns and df['title_cleaned'].sum() > 0:
        print(f"\n🔍 Sample of cleaned titles:")
        sample_cleaned = df[df['title_cleaned'] == True].head(3)
        for _, row in sample_cleaned.iterrows():
            print(f"  - '{row['title_original']}' → '{row['title']}'")
    
    # Performance note
    print(f"\n⚡ Performance: This version should be significantly faster than the standard version!")
    
    return output_path


if __name__ == "__main__":
    main()
