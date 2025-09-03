"""
Test script for Final CSV Builder
Tests the new implementation against the exact specifications.
"""

import sys
from pathlib import Path
import pandas as pd

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from data_processing.final_csv_builder import FinalCSVBuilder


def test_final_csv_builder():
    """Test the final CSV builder with a small sample."""
    print("🧪 Testing Final CSV Builder...")
    
    # Initialize the builder
    builder = FinalCSVBuilder()
    
    # Test with a small sample first
    print("\n📊 Testing with sample size 10...")
    output_path = builder.build_final_csv(sample_size=10)
    
    # Load and examine the output
    df = pd.read_csv(output_path)
    
    print(f"\n✅ Generated CSV: {output_path}")
    print(f"📊 Shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Verify required columns are present
    required_columns = [
        'work_id', 'book_id_list_en', 'title', 'publication_year', 
        'median_publication_year', 'language_codes_en', 'num_pages_median',
        'description', 'popular_shelves', 'author_id', 'author_name',
        'author_average_rating', 'author_ratings_count', 'series_id',
        'series_title', 'series_works_count', 'ratings_count_sum',
        'text_reviews_count_sum', 'average_rating_weighted_mean',
        'average_rating_weighted_median'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ Missing required columns: {missing_columns}")
        return False
    else:
        print("✅ All required columns present")
    
    # Verify data types and constraints
    print("\n🔍 Verifying data constraints...")
    
    # Check publication year range
    year_range_ok = df['publication_year'].between(2000, 2020, inclusive='both').all()
    print(f"📅 Publication years in 2000-2020: {year_range_ok}")
    
    # Check no empty work_ids or titles
    no_empty_work_ids = df['work_id'].notna().all()
    no_empty_titles = df['title'].notna().all()
    print(f"🆔 No empty work_ids: {no_empty_work_ids}")
    print(f"📖 No empty titles: {no_empty_titles}")
    
    # Check book_id_list_en is not empty
    non_empty_book_lists = df['book_id_list_en'].apply(lambda x: len(eval(x)) > 0).all()
    print(f"📚 All book_id_list_en non-empty: {non_empty_book_lists}")
    
    # Show sample data
    print("\n📋 Sample data:")
    print(df.head(3).to_string())
    
    # Show summary statistics
    print("\n📊 Summary statistics:")
    print(f"  - Publication year range: {df['publication_year'].min()} - {df['publication_year'].max()}")
    print(f"  - Average ratings: {df['average_rating_weighted_mean'].mean():.2f}")
    print(f"  - Total ratings sum: {df['ratings_count_sum'].sum():,}")
    print(f"  - Works with series: {df['series_id'].notna().sum()}")
    
    return True


def test_full_dataset():
    """Test with the full dataset (will take longer)."""
    print("\n🏗️ Testing with full dataset...")
    
    builder = FinalCSVBuilder()
    output_path = builder.build_final_csv()  # No sample size = full dataset
    
    print(f"\n✅ Full dataset CSV generated: {output_path}")
    
    # Load and show summary
    df = pd.read_csv(output_path)
    print(f"📊 Full dataset shape: {df.shape}")
    print(f"📅 Publication year range: {df['publication_year'].min()} - {df['publication_year'].max()}")
    print(f"📚 Total works: {len(df):,}")
    
    return output_path


if __name__ == "__main__":
    # Test with sample first
    if test_final_csv_builder():
        print("\n✅ Sample test passed!")
        
        # Ask if user wants to test full dataset
        response = input("\n🤔 Test with full dataset? (y/n): ")
        if response.lower() == 'y':
            test_full_dataset()
    else:
        print("\n❌ Sample test failed!")
