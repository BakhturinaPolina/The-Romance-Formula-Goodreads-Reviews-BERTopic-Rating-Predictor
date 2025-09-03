#!/usr/bin/env python3
"""
Test script for the improved data cleaning pipeline.
"""

import pandas as pd
import sys
import os
sys.path.append('.')

from data_cleaning_pipeline import RomanceNovelDataCleaner

def test_improved_pipeline():
    """Test the improved cleaning pipeline on the sample data."""
    
    print("🧪 Testing Improved Cleaning Pipeline")
    print("=" * 50)
    
    # Load the original sample
    sample_path = "data/processed/test_output/test_cleaned_sample.csv"
    if not os.path.exists(sample_path):
        print(f"❌ Sample file not found: {sample_path}")
        return
    
    # Load sample data (we'll use the original columns, not the cleaned ones)
    df = pd.read_csv(sample_path)
    
    # Remove the cleaned columns to test fresh
    columns_to_remove = [
        'title_original', 'title_cleaned', 'series_number_extracted',
        'author_name_normalized', 'author_potential_duplicate',
        'description_original', 'description_cleaned',
        'series_title_original', 'series_title_cleaned',
        'subgenre_contemporary_romance', 'subgenre_historical_romance',
        'subgenre_paranormal_romance', 'subgenre_romantic_suspense',
        'subgenre_romantic_fantasy', 'subgenre_science_fiction_romance',
        'subgenre_primary', 'data_quality_score'
    ]
    
    # Only remove columns that exist
    existing_columns = [col for col in columns_to_remove if col in df.columns]
    df = df.drop(columns=existing_columns)
    
    print(f"📊 Loaded sample with {len(df):,} records and {len(df.columns)} columns")
    
    # Save as test input
    test_input_path = "data/processed/test_output/test_input_for_improved.csv"
    df.to_csv(test_input_path, index=False)
    print(f"💾 Saved test input to: {test_input_path}")
    
    # Initialize improved cleaner
    cleaner = RomanceNovelDataCleaner(test_input_path, "data/processed/test_output")
    
    print("\n🔧 Running improved cleaning pipeline...")
    
    # Test individual functions
    print("\n1. Testing Title Cleaning...")
    df_cleaned = cleaner.clean_titles(df.copy())
    titles_modified = (df_cleaned['title_original'] != df_cleaned['title_cleaned']).sum()
    print(f"   Titles modified: {titles_modified:,} ({titles_modified/len(df)*100:.1f}%)")
    
    if titles_modified > 0:
        print("   Examples of title cleaning:")
        modified_titles = df_cleaned[df_cleaned['title_original'] != df_cleaned['title_cleaned']].head(3)
        for i, (_, row) in enumerate(modified_titles.iterrows()):
            print(f"     {i+1}. Original: \"{row['title_original']}\"")
            print(f"        Cleaned:  \"{row['title_cleaned']}\"")
            print()
    
    print("\n2. Testing Subgenre Classification...")
    df_cleaned = cleaner.classify_subgenres(df_cleaned)
    subgenre_cols = [col for col in df_cleaned.columns if col.startswith('subgenre_') and not col.endswith('_primary')]
    total_classifications = df_cleaned[subgenre_cols].sum().sum()
    books_classified = (df_cleaned[subgenre_cols].sum(axis=1) > 0).sum()
    print(f"   Total subgenre classifications: {total_classifications:,}")
    print(f"   Books classified: {books_classified:,} ({books_classified/len(df)*100:.1f}%)")
    
    if total_classifications > 0:
        print("   Subgenre breakdown:")
        for col in subgenre_cols:
            count = df_cleaned[col].sum()
            if count > 0:
                print(f"     {col}: {count:,} books")
    
    print("\n3. Testing Full Pipeline...")
    try:
        output_path = cleaner.run_cleaning_pipeline("test_improved_cleaned_sample.csv")
        print(f"   ✅ Full pipeline completed successfully!")
        print(f"   📁 Output saved to: {output_path}")
        
        # Load and validate results
        final_df = pd.read_csv(output_path)
        print(f"   📊 Final dataset: {len(final_df):,} records, {len(final_df.columns)} columns")
        
        # Check key improvements
        print("\n4. Validation Results:")
        print(f"   - Data quality score: {final_df['data_quality_score'].mean():.3f}")
        print(f"   - Series numbers extracted: {final_df['series_number_extracted'].notna().sum():,}")
        print(f"   - Author duplicates identified: {final_df['author_potential_duplicate'].sum():,}")
        
    except Exception as e:
        print(f"   ❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_improved_pipeline()
