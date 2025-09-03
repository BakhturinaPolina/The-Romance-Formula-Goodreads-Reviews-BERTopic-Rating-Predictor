#!/usr/bin/env python3
"""
Debug the quality filters implementation to find the discrepancy.
"""

import gzip
import json
import sys
sys.path.append('src')
from data_processing.quality_filters import QualityFilters

def debug_filters():
    """Debug the quality filters implementation."""
    print("=== DEBUGGING QUALITY FILTERS ===")
    
    # Load sample data
    with gzip.open('data/raw/goodreads_books_romance.json.gz', 'rt') as f:
        data = [json.loads(line) for line in f][:20]  # Test with 20 books
    
    # Create quality filters
    policy = {
        'quality_filters': {
            'publication_year_min': 2000,
            'language_code': 'eng',
            'ratings_count_min': 10,
            'description_completeness': 0.8
        }
    }
    
    filters = QualityFilters(policy)
    
    print(f"Testing individual filters on {len(data)} books:")
    
    for i, book in enumerate(data):
        print(f"\nBook {i+1}: {book.get('title', 'N/A')}")
        
        # Check each filter individually
        ratings_valid = filters._check_ratings_count(book)
        lang_valid = filters._check_language(book)
        year_valid = filters._check_publication_year(book)
        desc_valid = filters._check_description_completeness(book)
        
        print(f"  Ratings: {book.get('ratings_count')} (valid: {ratings_valid})")
        print(f"  Language: '{book.get('language_code')}' (valid: {lang_valid})")
        print(f"  Year: {book.get('publication_year')} (valid: {year_valid})")
        print(f"  Description: {len(book.get('description', ''))} chars (valid: {desc_valid})")
        
        # Check if all filters pass
        all_valid = ratings_valid and lang_valid and year_valid and desc_valid
        print(f"  ALL VALID: {all_valid}")
        
        if all_valid:
            print(f"  ✅ This book should pass all filters!")
        else:
            print(f"  ❌ This book fails at least one filter")

def compare_manual_vs_implementation():
    """Compare manual analysis vs implementation."""
    print("\n\n=== COMPARING MANUAL vs IMPLEMENTATION ===")
    
    # Load sample data
    with gzip.open('data/raw/goodreads_books_romance.json.gz', 'rt') as f:
        data = [json.loads(line) for line in f][:100]  # Test with 100 books
    
    # Manual analysis (like in analyze_filters.py)
    manual_passing = []
    for book in data:
        # Manual ratings check
        try:
            rating_count = int(book.get('ratings_count', 0))
            if rating_count < 10:
                continue
        except (ValueError, TypeError):
            continue
        
        # Manual language check
        lang = book.get('language_code', '')
        if lang not in ['eng', 'en-US', 'en-GB', 'en-CA', '']:
            continue
        
        # Manual year check
        year_str = book.get('publication_year', '')
        if not year_str or not year_str.strip():
            continue
        try:
            year = int(year_str)
            if year < 2000:
                continue
        except (ValueError, TypeError):
            continue
        
        # Manual description check
        desc = book.get('description', '')
        if not desc or not desc.strip():
            continue
        
        manual_passing.append(book)
    
    print(f"Manual analysis: {len(manual_passing)} books pass all filters")
    
    # Implementation analysis
    policy = {
        'quality_filters': {
            'publication_year_min': 2000,
            'language_code': 'eng',
            'ratings_count_min': 10,
            'description_completeness': 0.8
        }
    }
    
    filters = QualityFilters(policy)
    impl_passing = filters.filter_books(data)
    
    print(f"Implementation: {len(impl_passing)} books pass all filters")
    
    # Find discrepancies
    if len(manual_passing) != len(impl_passing):
        print(f"\nDISCREPANCY FOUND!")
        print(f"Manual: {len(manual_passing)}, Implementation: {len(impl_passing)}")
        
        # Check which books pass manual but fail implementation
        manual_ids = {book.get('book_id') for book in manual_passing}
        impl_ids = {book.get('book_id') for book in impl_passing}
        
        manual_only = manual_ids - impl_ids
        impl_only = impl_ids - manual_ids
        
        if manual_only:
            print(f"Books that pass manual but fail implementation: {len(manual_only)}")
            for book_id in list(manual_only)[:3]:
                book = next(b for b in data if b.get('book_id') == book_id)
                print(f"  - {book.get('title')} (ID: {book_id})")
        
        if impl_only:
            print(f"Books that pass implementation but fail manual: {len(impl_only)}")

if __name__ == "__main__":
    debug_filters()
    compare_manual_vs_implementation()
