"""
Run the Enhanced Final CSV Builder with comprehensive title handling.
This version includes fallback logic for missing titles and data quality validation.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from csv_building.final_csv_builder_working import OptimizedFinalCSVBuilder


def main():
    """Run the simplified CSV builder with essential romance novel data."""
    print("🚀 Running Simplified Final CSV Builder...")
    print("📖 Features:")
    print("  - Essential romance novel data only")
    print("  - Clean, simplified structure")
    print("  - Data quality validation")
    print("  - Performance optimized")
    
    # Initialize the enhanced builder
    builder = OptimizedFinalCSVBuilder()
    
    # Ask user for sample size or full dataset
    print("\n📊 Choose processing mode:")
    print("1. Test with sample (recommended for first run)")
    print("2. Process full dataset")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        try:
            sample_size = int(input("Enter sample size (e.g., 100): ").strip())
            print(f"\n📊 Processing sample of {sample_size} works...")
            output_path = builder.build_final_csv_optimized(sample_size=sample_size)
        except ValueError:
            print("Invalid sample size, using default of 100")
            output_path = builder.build_final_csv_optimized(sample_size=100)
    else:
        print("\n📊 Processing full dataset...")
        output_path = builder.build_final_csv_optimized()  # No sample size = full dataset
    
    print(f"\n✅ Simplified CSV with essential romance novel data generated: {output_path}")
    
    # Load and show summary
    import pandas as pd
    df = pd.read_csv(output_path)
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📅 Publication year range: {df['publication_year'].min()} - {df['publication_year'].max()}")
    print(f"📚 Total works: {len(df):,}")
    print(f"👤 Works with author data: {df['author_id'].notna().sum():,}")
    print(f"📖 Works with series data: {df['series_id'].notna().sum():,}")
    
    # Note: Title source and cleaning details removed for simplified output
    
    # Show data quality summary
    print(f"\n🔍 Data Quality Summary:")
    print(f"  - Works with titles: {len(df[df['title'] != 'Untitled']):,}")
    print(f"  - Works with 'Untitled': {(df['title'] == 'Untitled').sum():,}")
    print(f"  - Null descriptions: {df['description'].isnull().sum():,}")
    print(f"  - Null series info: {df['series_id'].isnull().sum():,}")
    
    return output_path


if __name__ == "__main__":
    main()
