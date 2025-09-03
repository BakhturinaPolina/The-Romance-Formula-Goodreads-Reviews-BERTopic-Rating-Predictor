"""
Analyze title columns in raw data to find the best source for clean titles.
"""

import json
import gzip
import pandas as pd
from pathlib import Path
from collections import Counter
import re

def analyze_title_columns():
    """Analyze title columns in raw data files."""
    
    print("🔍 Analyzing title columns in raw data...")
    
    # Analyze books data
    print("\n📚 Analyzing goodreads_books_romance.json.gz...")
    books_titles = []
    books_titles_without_series = []
    
    with gzip.open("data/raw/goodreads_books_romance.json.gz", 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 1000:  # Sample first 1000 records
                break
            if line.strip():
                try:
                    record = json.loads(line.strip())
                    if 'title' in record:
                        books_titles.append(record['title'])
                    if 'title_without_series' in record:
                        books_titles_without_series.append(record['title_without_series'])
                except json.JSONDecodeError:
                    continue
    
    print(f"📖 Sample titles from 'title' column:")
    for i, title in enumerate(books_titles[:10]):
        print(f"  {i+1}. {title}")
    
    print(f"\n📖 Sample titles from 'title_without_series' column:")
    for i, title in enumerate(books_titles_without_series[:10]):
        print(f"  {i+1}. {title}")
    
    # Analyze works data
    print(f"\n🔗 Analyzing goodreads_book_works.json.gz...")
    works_original_titles = []
    
    with gzip.open("data/raw/goodreads_book_works.json.gz", 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 1000:  # Sample first 1000 records
                break
            if line.strip():
                try:
                    record = json.loads(line.strip())
                    if 'original_title' in record:
                        works_original_titles.append(record['original_title'])
                except json.JSONDecodeError:
                    continue
    
    print(f"📖 Sample titles from 'original_title' column:")
    for i, title in enumerate(works_original_titles[:10]):
        print(f"  {i+1}. {title}")
    
    # Analyze quality metrics
    print(f"\n📊 Quality Analysis:")
    
    # Check for empty titles
    empty_books_title = sum(1 for t in books_titles if not t or t.strip() == "")
    empty_books_title_without_series = sum(1 for t in books_titles_without_series if not t or t.strip() == "")
    empty_works_original_title = sum(1 for t in works_original_titles if not t or t.strip() == "")
    
    print(f"  Empty 'title' in books: {empty_books_title}/{len(books_titles)} ({empty_books_title/len(books_titles)*100:.1f}%)")
    print(f"  Empty 'title_without_series' in books: {empty_books_title_without_series}/{len(books_titles_without_series)} ({empty_books_title_without_series/len(books_titles_without_series)*100:.1f}%)")
    print(f"  Empty 'original_title' in works: {empty_works_original_title}/{len(works_original_titles)} ({empty_works_original_title/len(works_original_titles)*100:.1f}%)")
    
    # Check for series indicators
    series_patterns = [
        r'\([^)]*#[0-9]+[^)]*\)',  # (Series Name, #1)
        r'#[0-9]+',  # #1, #2, etc.
        r'Book [0-9]+',  # Book 1, Book 2, etc.
        r'Volume [0-9]+',  # Volume 1, Volume 2, etc.
        r'Part [0-9]+',  # Part 1, Part 2, etc.
    ]
    
    def count_series_indicators(titles):
        count = 0
        for title in titles:
            if title:
                for pattern in series_patterns:
                    if re.search(pattern, title):
                        count += 1
                        break
        return count
    
    series_in_books_title = count_series_indicators(books_titles)
    series_in_books_title_without_series = count_series_indicators(books_titles_without_series)
    series_in_works_original_title = count_series_indicators(works_original_titles)
    
    print(f"\n  Series indicators in 'title': {series_in_books_title}/{len(books_titles)} ({series_in_books_title/len(books_titles)*100:.1f}%)")
    print(f"  Series indicators in 'title_without_series': {series_in_books_title_without_series}/{len(books_titles_without_series)} ({series_in_books_title_without_series/len(books_titles_without_series)*100:.1f}%)")
    print(f"  Series indicators in 'original_title': {series_in_works_original_title}/{len(works_original_titles)} ({series_in_works_original_title/len(works_original_titles)*100:.1f}%)")
    
    # Show examples of titles with series indicators
    print(f"\n🔍 Examples of titles with series indicators:")
    
    print(f"\n  From 'title' column:")
    series_examples = 0
    for title in books_titles:
        if series_examples >= 5:
            break
        for pattern in series_patterns:
            if re.search(pattern, title):
                print(f"    - {title}")
                series_examples += 1
                break
    
    print(f"\n  From 'title_without_series' column:")
    series_examples = 0
    for title in books_titles_without_series:
        if series_examples >= 5:
            break
        for pattern in series_patterns:
            if re.search(pattern, title):
                print(f"    - {title}")
                series_examples += 1
                break
    
    print(f"\n  From 'original_title' column:")
    series_examples = 0
    for title in works_original_titles:
        if series_examples >= 5:
            break
        for pattern in series_patterns:
            if re.search(pattern, title):
                print(f"    - {title}")
                series_examples += 1
                break
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    print(f"  1. 'title_without_series' appears to be the cleanest option")
    print(f"  2. 'original_title' from works data is also clean but may have more empty values")
    print(f"  3. 'title' from books data contains series information that needs cleaning")
    
    return {
        'books_titles': books_titles,
        'books_titles_without_series': books_titles_without_series,
        'works_original_titles': works_original_titles
    }

if __name__ == "__main__":
    analyze_title_columns()
