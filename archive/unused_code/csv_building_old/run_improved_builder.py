"""
Run the improved final CSV builder with title cleaning and better data quality.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from csv_building.final_csv_builder_improved import ImprovedFinalCSVBuilder


def main():
    """Run the improved final CSV builder."""
    print("🚀 Running Improved Final CSV Builder with Title Cleaning...")
    
    # Initialize the improved builder
    builder = ImprovedFinalCSVBuilder()
    
    # Ask user for sample size or full dataset
    print("\n📊 Choose processing mode:")
    print("1. Test with sample (recommended for first run)")
    print("2. Process full dataset")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        try:
            sample_size = int(input("Enter sample size (e.g., 1000): ").strip())
            print(f"\n📊 Processing sample of {sample_size} works...")
            output_path = builder.build_final_csv_improved(sample_size=sample_size)
        except ValueError:
            print("Invalid sample size, using default of 1000")
            output_path = builder.build_final_csv_improved(sample_size=1000)
    else:
        print("\n📊 Processing full dataset...")
        output_path = builder.build_final_csv_improved()  # No sample size = full dataset
    
    print(f"\n✅ Improved CSV generated: {output_path}")
    
    # Load and show summary
    import pandas as pd
    df = pd.read_csv(output_path)
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📅 Publication year range: {df['publication_year'].min()} - {df['publication_year'].max()}")
    print(f"📚 Total works: {len(df):,}")
    print(f"👤 Works with author data: {df['author_id'].notna().sum():,}")
    print(f"📖 Works with series data: {df['series_id'].notna().sum():,}")
    
    # Show title quality metrics
    if 'title_source' in df.columns:
        print(f"\n📖 Title Quality Summary:")
        title_source_counts = df['title_source'].value_counts()
        for source, count in title_source_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  - {source}: {count:,} ({percentage:.1f}%)")
    
    if 'title_has_series_info' in df.columns:
        titles_with_series = df['title_has_series_info'].sum()
        print(f"  - Titles with series info: {titles_with_series:,} ({titles_with_series/len(df)*100:.1f}%)")
    
    return output_path


if __name__ == "__main__":
    main()
