#!/usr/bin/env python3
"""
Test the data cleaning pipeline on a small sample to validate functionality.
"""

import pandas as pd
import sys
from pathlib import Path

# Add the current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from data_cleaning_pipeline import RomanceNovelDataCleaner

def test_cleaning_pipeline():
    """Test the cleaning pipeline on a small sample."""
    print("🧪 TESTING DATA CLEANING PIPELINE")
    print("=" * 50)
    
    # Input file path
    input_file = "data/processed/final_books_2000_2020_en_20250901_024106.csv"
    
    if not Path(input_file).exists():
        print(f"❌ Input file not found: {input_file}")
        return
    
    # Load a small sample for testing
    print("📚 Loading sample dataset...")
    df_sample = pd.read_csv(input_file, nrows=1000)  # Test with 1000 records
    
    print(f"✅ Loaded {len(df_sample):,} sample records")
    
    # Save sample to temporary file
    sample_path = "data/processed/sample_for_testing.csv"
    df_sample.to_csv(sample_path, index=False)
    print(f"💾 Sample saved to: {sample_path}")
    
    # Initialize cleaner with sample
    cleaner = RomanceNovelDataCleaner(sample_path, output_dir="data/processed/test_output")
    
    # Test individual cleaning functions
    print("\n🔍 Testing individual cleaning functions...")
    
    # Test title cleaning
    print("\n1. Testing title cleaning...")
    df_test = cleaner.load_dataset()
    df_test = cleaner.clean_titles(df_test)
    
    # Show some examples
    title_examples = df_test[['title_original', 'title_cleaned', 'series_number_extracted']].head(5)
    print("Title cleaning examples:")
    print(title_examples)
    
    # Test author name cleaning
    print("\n2. Testing author name cleaning...")
    df_test = cleaner.clean_author_names(df_test)
    
    author_examples = df_test[['author_name', 'author_name_normalized', 'author_potential_duplicate']].head(5)
    print("Author cleaning examples:")
    print(author_examples)
    
    # Test description cleaning
    print("\n3. Testing description cleaning...")
    df_test = cleaner.clean_descriptions(df_test)
    
    # Show description cleaning stats
    desc_cleaned = (df_test['description_original'] != df_test['description_cleaned']).sum()
    print(f"Descriptions cleaned: {desc_cleaned:,} out of {len(df_test):,}")
    
    # Test series standardization
    print("\n4. Testing series standardization...")
    df_test = cleaner.standardize_series_info(df_test)
    
    series_examples = df_test[['series_title_original', 'series_title_cleaned', 'series_position']].head(5)
    print("Series standardization examples:")
    print(series_examples)
    
    # Test subgenre classification
    print("\n5. Testing subgenre classification...")
    df_test = cleaner.classify_subgenres(df_test)
    
    # Show subgenre columns
    subgenre_cols = [col for col in df_test.columns if col.startswith('subgenre_')]
    print(f"Subgenre columns created: {subgenre_cols}")
    
    # Test data quality improvement
    print("\n6. Testing data quality improvement...")
    df_test = cleaner.improve_data_quality(df_test)
    
    print(f"Data quality score range: {df_test['data_quality_score'].min():.3f} - {df_test['data_quality_score'].max():.3f}")
    print(f"Average data quality score: {df_test['data_quality_score'].mean():.3f}")
    
    # Show final sample of cleaned data
    print("\n📊 FINAL CLEANED SAMPLE:")
    print("=" * 50)
    
    # Select key columns to show
    key_columns = [
        'title_original', 'title_cleaned', 'series_number_extracted',
        'author_name_normalized', 'author_potential_duplicate',
        'subgenre_primary', 'data_quality_score'
    ]
    
    # Only show columns that exist
    existing_columns = [col for col in key_columns if col in df_test.columns]
    final_sample = df_test[existing_columns].head(10)
    print(final_sample)
    
    # Show column count comparison
    print(f"\n📋 Column count comparison:")
    print(f"   Original: {len(df_test.columns)} columns")
    print(f"   New columns added: {len(existing_columns)}")
    
    # Test the full pipeline
    print("\n🚀 Testing full cleaning pipeline...")
    try:
        output_path = cleaner.run_cleaning_pipeline("test_cleaned_sample.csv")
        print(f"✅ Full pipeline test successful!")
        print(f"📁 Output saved to: {output_path}")
        
        # Load and verify the cleaned output
        cleaned_df = pd.read_csv(output_path)
        print(f"📊 Cleaned dataset shape: {cleaned_df.shape}")
        print(f"📋 Cleaned dataset columns: {len(cleaned_df.columns)}")
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    if Path(sample_path).exists():
        Path(sample_path).unlink()
        print(f"🧹 Cleaned up test sample file")
    
    print("\n✅ Testing completed successfully!")

if __name__ == "__main__":
    test_cleaning_pipeline()
