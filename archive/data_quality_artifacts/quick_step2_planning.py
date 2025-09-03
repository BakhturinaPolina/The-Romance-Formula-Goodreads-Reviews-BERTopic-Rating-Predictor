#!/usr/bin/env python3
"""
Quick Step 2 Planning Analysis
Get key insights for duplicate detection and inconsistency analysis.
"""

import pandas as pd
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from data_quality.data_quality_assessment import DataQualityAssessment


def quick_step2_analysis():
    """Quick analysis for Step 2 planning."""
    print("🔍 Quick Step 2 Planning Analysis")
    print("=" * 50)
    
    # Load the cleaned dataset
    cleaned_files = list(Path("../../data/processed").glob("final_books_*_cleaned_nlp_ready_*.csv"))
    if not cleaned_files:
        print("❌ No cleaned dataset found")
        return
    
    cleaned_file = cleaned_files[0]
    print(f"📚 Loading: {cleaned_file.name}")
    
    try:
        data = pd.read_csv(cleaned_file)
        print(f"✅ Loaded: {len(data):,} books")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return
    
    print(f"\n📊 KEY INSIGHTS FOR STEP 2:")
    
    # 1. Work ID analysis
    work_id_duplicates = data['work_id'].duplicated().sum()
    print(f"  📚 Work ID Duplicates: {work_id_duplicates:,}")
    
    # 2. Title analysis
    title_duplicates = data['title'].duplicated().sum()
    title_unique = data['title'].nunique()
    print(f"  📖 Title Duplicates: {title_duplicates:,} ({title_duplicates/len(data)*100:.1f}%)")
    print(f"    - Unique titles: {title_unique:,}")
    
    if title_duplicates > 0:
        duplicate_titles = data[data['title'].duplicated(keep=False)]['title'].value_counts().head(5)
        print(f"    - Top duplicates: {duplicate_titles.to_dict()}")
    
    # 3. Author analysis
    author_unique = data['author_id'].nunique()
    avg_books_per_author = len(data) / author_unique
    print(f"  👤 Authors: {author_unique:,} unique")
    print(f"    - Avg books per author: {avg_books_per_author:.1f}")
    
    # 4. Series analysis
    series_books = data['series_id'].notna().sum()
    standalone_books = len(data) - series_books
    print(f"  📚 Series: {series_books:,} books in series")
    print(f"    - Standalone: {standalone_books:,} books")
    
    # 5. Genre analysis
    genre_unique = data['genres'].nunique()
    print(f"  🏷️  Genres: {genre_unique:,} unique combinations")
    
    # 6. Publication year
    year_range = f"{data['publication_year'].min()} - {data['publication_year'].max()}"
    print(f"  📅 Publication Years: {year_range}")
    
    # 7. Rating analysis
    rating_null = data['average_rating_weighted_mean'].isnull().sum()
    print(f"  ⭐ Ratings: {rating_null:,} missing ({rating_null/len(data)*100:.1f}%)")
    
    print(f"\n🎯 STEP 2 IMPLEMENTATION PLAN:")
    print(f"  Based on this analysis:")
    
    if work_id_duplicates > 0:
        print(f"  1. 🔴 CRITICAL: Fix {work_id_duplicates} duplicate work IDs")
    else:
        print(f"  1. ✅ Work IDs are unique - no action needed")
    
    if title_duplicates > 0:
        print(f"  2. 🟡 HIGH: Investigate {title_duplicates} duplicate titles")
        print(f"     - Focus on common titles like 'Broken', 'Second Chances'")
        print(f"     - Check if these are legitimate duplicates or different books")
    
    print(f"  3. 🟡 HIGH: Series data validation")
    print(f"     - {series_books:,} books claim to be in series")
    print(f"     - Validate series_works_count vs. actual series size")
    
    print(f"  4. 🟢 MEDIUM: Author consistency checks")
    print(f"     - {author_unique:,} unique authors for {len(data):,} books")
    print(f"     - Verify author attribution accuracy")
    
    print(f"  5. 🟢 MEDIUM: Genre classification validation")
    print(f"     - {genre_unique:,} unique genre combinations")
    print(f"     - Check for inconsistent genre patterns")
    
    print(f"  6. 🟢 MEDIUM: Publication year validation")
    print(f"     - Ensure all years are within 2000-2020 range")
    
    print(f"\n📋 NEXT ACTIONS:")
    print(f"  1. Implement duplicate detection algorithms")
    print(f"  2. Create inconsistency validation checks")
    print(f"  3. Generate detailed quality report")
    print(f"  4. Plan data cleaning strategies")
    
    return data


if __name__ == "__main__":
    data = quick_step2_analysis()
    
    if data is not None:
        print(f"\n✅ Quick analysis completed!")
        print(f"📝 Ready to implement Step 2: Duplicate and Inconsistency Detection")
    else:
        print(f"\n❌ Analysis failed!")
        sys.exit(1)
