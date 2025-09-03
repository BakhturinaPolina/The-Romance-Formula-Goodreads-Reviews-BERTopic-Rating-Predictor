"""
Run the cleaned titles CSV builder that uses only works.original_title with bracket stripping.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from csv_building.final_csv_builder_cleaned_titles import FinalCSVBuilderCleanedTitles


def main():
    """Run the cleaned titles CSV builder."""
    print("🚀 Running Final CSV Builder with Cleaned Titles...")
    print("📖 Uses ONLY works.original_title with simple bracket stripping")
    
    # Initialize the cleaned titles builder
    builder = FinalCSVBuilderCleanedTitles()
    
    # Ask user for sample size or full dataset
    print("\n📊 Choose processing mode:")
    print("1. Test with sample (recommended for first run)")
    print("2. Process full dataset")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        try:
            sample_size = int(input("Enter sample size (e.g., 1000): ").strip())
            print(f"\n📊 Processing sample of {sample_size} works...")
            output_path = builder.build_final_csv_cleaned_titles(sample_size=sample_size)
        except ValueError:
            print("Invalid sample size, using default of 1000")
            output_path = builder.build_final_csv_cleaned_titles(sample_size=1000)
    else:
        print("\n📊 Processing full dataset...")
        output_path = builder.build_final_csv_cleaned_titles()  # No sample size = full dataset
    
    print(f"\n✅ Cleaned titles CSV generated: {output_path}")
    
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
    
    return output_path


if __name__ == "__main__":
    main()
