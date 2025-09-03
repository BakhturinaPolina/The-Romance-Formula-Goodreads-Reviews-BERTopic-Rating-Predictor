#!/usr/bin/env python3
"""
Analyze why quality filters are filtering out all books.
"""

import gzip
import json

def analyze_filter_progression():
    """Analyze how many books pass each filter step."""
    print("=== FILTER PROGRESSION ANALYSIS ===")
    
    # Load all books data
    print("Loading books data...")
    with gzip.open('data/raw/goodreads_books_romance.json.gz', 'rt') as f:
        data = [json.loads(line) for line in f]
    
    print(f"Total books: {len(data):,}")
    
    # Step 1: Ratings count >= 10
    ratings_10 = []
    for book in data:
        try:
            rating_count = int(book.get('ratings_count', 0))
            if rating_count >= 10:
                ratings_10.append(book)
        except (ValueError, TypeError):
            continue
    
    print(f"Books with >= 10 ratings: {len(ratings_10):,}")
    
    # Step 2: Language filter (English or empty)
    lang_eng = []
    for book in ratings_10:
        lang = book.get('language_code', '')
        if lang in ['eng', 'en-US', 'en-GB', 'en-CA', '']:
            lang_eng.append(book)
    
    print(f"Books with English/empty language: {len(lang_eng):,}")
    
    # Step 3: Publication year >= 2000
    year_2000 = []
    for book in lang_eng:
        year_str = book.get('publication_year', '')
        if year_str and year_str.strip():
            try:
                year = int(year_str)
                if year >= 2000:
                    year_2000.append(book)
            except (ValueError, TypeError):
                continue
    
    print(f"Books with year >= 2000: {len(year_2000):,}")
    
    # Step 4: Has description
    with_desc = []
    for book in year_2000:
        desc = book.get('description', '')
        if desc and desc.strip():
            with_desc.append(book)
    
    print(f"Books with description: {len(with_desc):,}")
    
    print(f"\nFinal result: {len(with_desc):,} books pass all filters")
    
    if with_desc:
        print(f"\nSample passing book:")
        sample = with_desc[0]
        print(f"  Title: {sample.get('title', 'N/A')}")
        print(f"  Book ID: {sample.get('book_id', 'N/A')}")
        print(f"  Ratings: {sample.get('ratings_count', 'N/A')}")
        print(f"  Language: '{sample.get('language_code', 'N/A')}'")
        print(f"  Year: {sample.get('publication_year', 'N/A')}")
        print(f"  Description length: {len(sample.get('description', ''))}")
    
    return len(with_desc)

def check_quality_filters_implementation():
    """Check if our quality filters implementation matches the analysis."""
    print("\n=== QUALITY FILTERS IMPLEMENTATION CHECK ===")
    
    # Import our quality filters
    import sys
    sys.path.append('src')
    from data_processing.quality_filters import QualityFilters
    
    # Load sample data
    with gzip.open('data/raw/goodreads_books_romance.json.gz', 'rt') as f:
        sample_data = [json.loads(line) for line in f][:1000]  # Test with 1000 books
    
    print(f"Testing with {len(sample_data)} sample books...")
    
    # Create quality filters
    sampling_policy = {
        'quality_filters': {
            'publication_year_min': 2000,
            'language_code': 'eng',
            'ratings_count_min': 10,
            'description_completeness': 0.8
        }
    }
    
    filters = QualityFilters(sampling_policy)
    
    # Apply filters
    filtered = filters.filter_books(sample_data)
    print(f"Books passing our filters: {len(filtered)}")
    
    if len(filtered) == 0:
        print("\nDEBUGGING: Let's check each filter individually...")
        
        # Test each filter separately
        for i, book in enumerate(sample_data[:5]):
            print(f"\nBook {i+1}: {book.get('title', 'N/A')}")
            
            # Check publication year
            year_valid = filters._check_publication_year(book)
            print(f"  Publication year valid: {year_valid} (year: {book.get('publication_year', 'N/A')})")
            
            # Check language
            lang_valid = filters._check_language(book)
            print(f"  Language valid: {lang_valid} (lang: '{book.get('language_code', 'N/A')}')")
            
            # Check ratings
            ratings_valid = filters._check_ratings_count(book)
            print(f"  Ratings valid: {ratings_valid} (ratings: {book.get('ratings_count', 'N/A')})")
            
            # Check description
            desc_valid = filters._check_description_completeness(book)
            print(f"  Description valid: {desc_valid} (desc length: {len(book.get('description', ''))})")

if __name__ == "__main__":
    analyze_filter_progression()
    check_quality_filters_implementation()
