#!/usr/bin/env python3
"""
Systematic analysis of raw data files to understand data types and quality.
"""

import gzip
import json
from pathlib import Path
from collections import Counter

def analyze_books_data():
    """Analyze the books data file."""
    print("=== BOOKS DATA ANALYSIS ===")
    
    # Load sample data
    books_data = []
    with gzip.open('data/raw/goodreads_books_romance.json.gz', 'rt') as f:
        for i, line in enumerate(f):
            if i >= 100:  # Sample 100 books
                break
            books_data.append(json.loads(line))
    
    print(f"Analyzed {len(books_data)} sample books")
    
    # Check fields
    all_fields = set()
    for book in books_data:
        all_fields.update(book.keys())
    
    print(f"\nAll fields found: {sorted(all_fields)}")
    
    # Analyze key fields
    key_fields = ['book_id', 'work_id', 'ratings_count', 'text_reviews_count', 
                  'average_rating', 'publication_year', 'language_code', 'num_pages']
    
    print("\n=== KEY FIELDS ANALYSIS ===")
    for field in key_fields:
        if field not in all_fields:
            print(f"{field}: FIELD NOT FOUND")
            continue
            
        values = [book.get(field) for book in books_data]
        types = [type(v).__name__ for v in values]
        type_counts = Counter(types)
        
        print(f"\n{field}:")
        print(f"  Types: {dict(type_counts)}")
        print(f"  Sample values: {values[:5]}")
        
        # Check for empty/null values
        empty_count = sum(1 for v in values if v is None or v == '' or v == 'None')
        print(f"  Empty/null values: {empty_count}/{len(values)} ({empty_count/len(values)*100:.1f}%)")
        
        # Check for non-empty values
        non_empty = [v for v in values if v is not None and v != '' and v != 'None']
        if non_empty:
            print(f"  Non-empty sample: {non_empty[:3]}")
    
    # Check language codes specifically
    print("\n=== LANGUAGE CODE ANALYSIS ===")
    language_codes = [book.get('language_code', '') for book in books_data]
    lang_counts = Counter(language_codes)
    print(f"Language code distribution: {dict(lang_counts)}")
    
    # Check publication years
    print("\n=== PUBLICATION YEAR ANALYSIS ===")
    years = []
    for book in books_data:
        year = book.get('publication_year', '')
        if year and year != '' and year != 'None':
            try:
                years.append(int(year))
            except (ValueError, TypeError):
                print(f"Invalid year: {year}")
    
    if years:
        print(f"Year range: {min(years)} - {max(years)}")
        print(f"Years >= 2000: {sum(1 for y in years if y >= 2000)}/{len(years)}")
    
    # Check ratings
    print("\n=== RATINGS ANALYSIS ===")
    ratings = []
    for book in books_data:
        rating = book.get('ratings_count', '')
        if rating and rating != '' and rating != 'None':
            try:
                ratings.append(int(rating))
            except (ValueError, TypeError):
                print(f"Invalid rating count: {rating}")
    
    if ratings:
        print(f"Rating count range: {min(ratings)} - {max(ratings)}")
        print(f"Books with >= 10 ratings: {sum(1 for r in ratings if r >= 10)}/{len(ratings)}")

def analyze_reviews_data():
    """Analyze the reviews data file."""
    print("\n\n=== REVIEWS DATA ANALYSIS ===")
    
    # Load sample data
    reviews_data = []
    with gzip.open('data/raw/goodreads_reviews_romance.json.gz', 'rt') as f:
        for i, line in enumerate(f):
            if i >= 100:  # Sample 100 reviews
                break
            reviews_data.append(json.loads(line))
    
    print(f"Analyzed {len(reviews_data)} sample reviews")
    
    # Check fields
    all_fields = set()
    for review in reviews_data:
        all_fields.update(review.keys())
    
    print(f"\nAll fields found: {sorted(all_fields)}")
    
    # Analyze key fields
    key_fields = ['book_id', 'user_id', 'rating', 'review_text', 'review_date']
    
    print("\n=== KEY FIELDS ANALYSIS ===")
    for field in key_fields:
        if field not in all_fields:
            print(f"{field}: FIELD NOT FOUND")
            continue
            
        values = [review.get(field) for review in reviews_data]
        types = [type(v).__name__ for v in values]
        type_counts = Counter(types)
        
        print(f"\n{field}:")
        print(f"  Types: {dict(type_counts)}")
        print(f"  Sample values: {values[:3]}")
        
        # Check for empty/null values
        empty_count = sum(1 for v in values if v is None or v == '' or v == 'None')
        print(f"  Empty/null values: {empty_count}/{len(values)} ({empty_count/len(values)*100:.1f}%)")

def analyze_authors_data():
    """Analyze the authors data file."""
    print("\n\n=== AUTHORS DATA ANALYSIS ===")
    
    # Load sample data
    authors_data = []
    with gzip.open('data/raw/goodreads_book_authors.json.gz', 'rt') as f:
        for i, line in enumerate(f):
            if i >= 100:  # Sample 100 author records
                break
            authors_data.append(json.loads(line))
    
    print(f"Analyzed {len(authors_data)} sample author records")
    
    # Check fields
    all_fields = set()
    for author in authors_data:
        all_fields.update(author.keys())
    
    print(f"\nAll fields found: {sorted(all_fields)}")

def analyze_works_data():
    """Analyze the works data file."""
    print("\n\n=== WORKS DATA ANALYSIS ===")
    
    # Load sample data
    works_data = []
    with gzip.open('data/raw/goodreads_book_works.json.gz', 'rt') as f:
        for i, line in enumerate(f):
            if i >= 100:  # Sample 100 work records
                break
            works_data.append(json.loads(line))
    
    print(f"Analyzed {len(works_data)} sample work records")
    
    # Check fields
    all_fields = set()
    for work in works_data:
        all_fields.update(work.keys())
    
    print(f"\nAll fields found: {sorted(all_fields)}")

if __name__ == "__main__":
    analyze_books_data()
    analyze_reviews_data()
    analyze_authors_data()
    analyze_works_data()
