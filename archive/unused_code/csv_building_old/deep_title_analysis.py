"""
Deep analysis of title columns to understand why title_without_series still contains series info.
"""

import json
import gzip
import pandas as pd
from pathlib import Path
import re

def deep_title_analysis():
    """Perform deep analysis of title columns."""
    
    print("🔍 Deep Analysis of Title Columns...")
    
    # Load sample data for detailed inspection
    books_sample = []
    works_sample = []
    
    # Sample from books data
    print("\n📚 Sampling from goodreads_books_romance.json.gz...")
    with gzip.open("data/raw/goodreads_books_romance.json.gz", 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 200:  # Sample first 200 records
                break
            if line.strip():
                try:
                    record = json.loads(line.strip())
                    books_sample.append(record)
                except json.JSONDecodeError:
                    continue
    
    # Sample from works data
    print("🔗 Sampling from goodreads_book_works.json.gz...")
    with gzip.open("data/raw/goodreads_book_works.json.gz", 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 200:  # Sample first 200 records
                break
            if line.strip():
                try:
                    record = json.loads(line.strip())
                    works_sample.append(record)
                except json.JSONDecodeError:
                    continue
    
    print(f"📊 Sampled {len(books_sample)} books and {len(works_sample)} works")
    
    # Check all available columns in books data
    print(f"\n🔍 Available columns in books data:")
    if books_sample:
        all_columns = set()
        for record in books_sample:
            all_columns.update(record.keys())
        
        title_related_columns = [col for col in sorted(all_columns) if 'title' in col.lower()]
        print(f"  Title-related columns: {title_related_columns}")
        
        # Show all columns for context
        print(f"  All columns: {sorted(all_columns)}")
    
    # Check all available columns in works data
    print(f"\n🔍 Available columns in works data:")
    if works_sample:
        all_columns = set()
        for record in works_sample:
            all_columns.update(record.keys())
        
        title_related_columns = [col for col in sorted(all_columns) if 'title' in col.lower()]
        print(f"  Title-related columns: {title_related_columns}")
    
    # Detailed comparison of title columns
    print(f"\n📖 Detailed Title Comparison:")
    
    # Find records that have both title and title_without_series
    comparison_records = []
    for record in books_sample:
        if 'title' in record and 'title_without_series' in record:
            comparison_records.append(record)
    
    print(f"  Records with both columns: {len(comparison_records)}")
    
    # Show examples where titles differ
    different_titles = []
    for record in comparison_records[:20]:  # First 20
        title = record.get('title', '')
        title_without_series = record.get('title_without_series', '')
        
        if title != title_without_series:
            different_titles.append({
                'title': title,
                'title_without_series': title_without_series,
                'book_id': record.get('book_id', ''),
                'work_id': record.get('work_id', '')
            })
    
    print(f"\n  Titles that differ between columns: {len(different_titles)}")
    for i, diff in enumerate(different_titles[:10]):
        print(f"    {i+1}. Book ID: {diff['book_id']}, Work ID: {diff['work_id']}")
        print(f"       'title': {diff['title']}")
        print(f"       'title_without_series': {diff['title_without_series']}")
        print()
    
    # Check if title_without_series is actually different from title
    identical_count = sum(1 for record in comparison_records 
                         if record.get('title') == record.get('title_without_series'))
    
    print(f"  Identical titles: {identical_count}/{len(comparison_records)} ({identical_count/len(comparison_records)*100:.1f}%)")
    
    # Look for patterns in series information
    print(f"\n🔍 Series Information Patterns:")
    
    series_patterns = [
        r'\([^)]*#[0-9]+[^)]*\)',  # (Series Name, #1)
        r'#[0-9]+',  # #1, #2, etc.
        r'Book [0-9]+',  # Book 1, Book 2, etc.
        r'Volume [0-9]+',  # Volume 1, Volume 2, etc.
        r'Part [0-9]+',  # Part 1, Part 2, etc.
        r'Novella',  # Novella
        r'Series',  # Series
    ]
    
    def extract_series_info(title):
        """Extract series information from title."""
        series_info = []
        for pattern in series_patterns:
            matches = re.findall(pattern, title, re.IGNORECASE)
            if matches:
                series_info.extend(matches)
        return series_info
    
    # Analyze series patterns in both columns
    title_series_count = 0
    title_without_series_count = 0
    
    for record in comparison_records:
        title = record.get('title', '')
        title_without_series = record.get('title_without_series', '')
        
        title_series = extract_series_info(title)
        title_without_series_series = extract_series_info(title_without_series)
        
        if title_series:
            title_series_count += 1
        if title_without_series_series:
            title_without_series_count += 1
    
    print(f"  'title' with series info: {title_series_count}/{len(comparison_records)} ({title_series_count/len(comparison_records)*100:.1f}%)")
    print(f"  'title_without_series' with series info: {title_without_series_count}/{len(comparison_records)} ({title_without_series_count/len(comparison_records)*100:.1f}%)")
    
    # Check if there are other title columns
    print(f"\n🔍 Looking for alternative title sources...")
    
    # Check if there's a 'name' column or similar
    name_columns = [col for col in all_columns if 'name' in col.lower() and col != 'author_name']
    if name_columns:
        print(f"  Found name-related columns: {name_columns}")
        
        # Sample from name columns
        for col in name_columns[:3]:  # Check first 3
            values = [record.get(col, '') for record in books_sample if record.get(col)]
            non_empty = [v for v in values if v and str(v).strip()]
            print(f"    '{col}': {len(non_empty)} non-empty values")
            if non_empty:
                print(f"      Sample: {non_empty[:3]}")
    
    # Check works data for alternative titles
    print(f"\n🔍 Checking works data for alternative titles...")
    
    works_title_columns = []
    if works_sample:
        works_all_columns = set()
        for record in works_sample:
            works_all_columns.update(record.keys())
        
        works_title_columns = [col for col in sorted(works_all_columns) if 'title' in col.lower()]
        print(f"  Works title columns: {works_title_columns}")
        
        # Check if there are other title-like fields
        title_like_columns = [col for col in works_all_columns if any(word in col.lower() for word in ['title', 'name', 'book'])]
        print(f"  Title-like columns in works: {title_like_columns}")
    
    # Final recommendations
    print(f"\n💡 Final Recommendations:")
    
    if title_without_series_count == title_series_count:
        print(f"  1. ⚠️  'title_without_series' is NOT actually clean - it's identical to 'title'")
        print(f"  2. 🔍 Need to find or create a truly clean title column")
        print(f"  3. 📝 Consider using 'original_title' from works data (but has {len([w for w in works_sample if not w.get('original_title')])/len(works_sample)*100:.1f}% empty values)")
    
    # Check if we can create a clean title by combining sources
    print(f"\n🔧 Potential Solutions:")
    print(f"  1. Use 'original_title' from works data as primary source")
    print(f"  2. Fall back to 'title' from books data when works data is empty")
    print(f"  3. Implement series removal logic for titles that still contain series info")
    
    return {
        'books_sample': books_sample,
        'works_sample': works_sample,
        'comparison_records': comparison_records,
        'different_titles': different_titles
    }

if __name__ == "__main__":
    deep_title_analysis()
