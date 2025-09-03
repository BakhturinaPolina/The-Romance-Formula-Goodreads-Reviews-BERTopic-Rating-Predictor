"""
Run the optimized final CSV builder with the full dataset.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from data_processing.final_csv_builder_optimized import OptimizedFinalCSVBuilder


def main():
    """Run the optimized final CSV builder."""
    print("🚀 Running Optimized Final CSV Builder with Full Dataset...")
    
    # Initialize the optimized builder
    builder = OptimizedFinalCSVBuilder()
    
    # Run with full dataset (no sample size)
    print("\n📊 Processing full dataset...")
    output_path = builder.build_final_csv_optimized()  # No sample size = full dataset
    
    print(f"\n✅ Full dataset CSV generated: {output_path}")
    
    # Load and show summary
    import pandas as pd
    df = pd.read_csv(output_path)
    print(f"📊 Full dataset shape: {df.shape}")
    print(f"📅 Publication year range: {df['publication_year'].min()} - {df['publication_year'].max()}")
    print(f"📚 Total works: {len(df):,}")
    print(f"👤 Works with author data: {df['author_id'].notna().sum():,}")
    print(f"📖 Works with series data: {df['series_id'].notna().sum():,}")
    
    return output_path


if __name__ == "__main__":
    main()
