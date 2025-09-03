#!/usr/bin/env python3
"""
Test Author Balancing Logic
Check if author balancing is causing the 0 books issue.
"""

import gzip
import json
import sys
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_processing.config_loader import ConfigLoader
from data_processing.quality_filters import QualityFilters, AuthorBalancer, DecadeStratifier


def setup_logging():
    """Set up logging for analysis."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'logs/validation/test_author_balancing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger(__name__)


def load_and_convert_sample_data(sample_size: int = 1000) -> List[Dict[str, Any]]:
    """Load sample data and convert data types."""
    logger = logging.getLogger(__name__)
    
    data_path = Path('data/raw/goodreads_books_romance.json.gz')
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading {sample_size} sample books...")
    
    with gzip.open(data_path, 'rt') as f:
        raw_books = []
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            raw_books.append(json.loads(line))
    
    # Convert data types
    converted_books = []
    numeric_conversions = {
        'text_reviews_count': int,
        'ratings_count': int,
        'num_pages': int,
        'publication_day': int,
        'publication_month': int,
        'publication_year': int,
        'book_id': int,
        'work_id': int,
        'average_rating': float,
    }
    
    for book in raw_books:
        converted_book = book.copy()
        for field, conversion_func in numeric_conversions.items():
            if field in converted_book:
                try:
                    value = converted_book[field]
                    if value is not None and value != '':
                        converted_book[field] = conversion_func(value)
                except (ValueError, TypeError):
                    pass
        converted_books.append(converted_book)
    
    return converted_books


def analyze_author_data_structure(books: List[Dict[str, Any]]):
    """Analyze the structure of author data in books."""
    logger = logging.getLogger(__name__)
    
    logger.info("=== ANALYZING AUTHOR DATA STRUCTURE ===")
    
    # Check for author_id field
    books_with_author_id = 0
    books_with_authors_field = 0
    books_with_author_name = 0
    
    sample_author_structures = []
    
    for book in books[:100]:  # Check first 100 books
        has_author_id = 'author_id' in book and book['author_id'] is not None
        has_authors_field = 'authors' in book and book['authors'] is not None
        has_author_name = 'author_name' in book and book['author_name'] is not None
        
        if has_author_id:
            books_with_author_id += 1
        if has_authors_field:
            books_with_authors_field += 1
        if has_author_name:
            books_with_author_name += 1
        
        # Collect sample structures
        if has_authors_field and len(sample_author_structures) < 5:
            sample_author_structures.append({
                'book_id': book.get('book_id'),
                'authors': book.get('authors'),
                'author_id': book.get('author_id'),
                'author_name': book.get('author_name')
            })
    
    logger.info(f"Books with author_id field: {books_with_author_id}/100")
    logger.info(f"Books with authors field: {books_with_authors_field}/100")
    logger.info(f"Books with author_name field: {books_with_author_name}/100")
    
    logger.info("Sample author structures:")
    for i, sample in enumerate(sample_author_structures):
        logger.info(f"  Sample {i+1}: {sample}")


def test_author_balancing_step_by_step(books: List[Dict[str, Any]], sampling_policy: Dict[str, Any]):
    """Test author balancing step by step."""
    logger = logging.getLogger(__name__)
    
    logger.info("=== TESTING AUTHOR BALANCING STEP BY STEP ===")
    
    # Step 1: Apply basic quality filters
    logger.info("Step 1: Applying basic quality filters...")
    quality_filter = QualityFilters(sampling_policy)
    filtered_books = quality_filter.filter_books(books)
    logger.info(f"Books after quality filters: {len(filtered_books)}")
    
    if len(filtered_books) == 0:
        logger.error("No books passed quality filters!")
        return
    
    # Step 2: Test author balancing
    logger.info("Step 2: Testing author balancing...")
    author_balancer = AuthorBalancer(sampling_policy)
    
    # Test author_id extraction
    logger.info("Testing author_id extraction...")
    author_ids_found = 0
    for book in filtered_books[:50]:  # Test first 50 books
        author_id = author_balancer._extract_author_id(book)
        if author_id:
            author_ids_found += 1
    
    logger.info(f"Author IDs found: {author_ids_found}/50")
    
    # Apply author balancing
    balanced_books = author_balancer.balance_authors(filtered_books)
    logger.info(f"Books after author balancing: {len(balanced_books)}")
    
    if len(balanced_books) == 0:
        logger.error("No books after author balancing!")
        return
    
    # Step 3: Test decade stratification
    logger.info("Step 3: Testing decade stratification...")
    decade_stratifier = DecadeStratifier(sampling_policy)
    final_books = decade_stratifier.stratify_by_decade(balanced_books)
    logger.info(f"Books after decade stratification: {len(final_books)}")
    
    # Summary
    logger.info("=== SUMMARY ===")
    logger.info(f"Original books: {len(books)}")
    logger.info(f"After quality filters: {len(filtered_books)}")
    logger.info(f"After author balancing: {len(balanced_books)}")
    logger.info(f"After decade stratification: {len(final_books)}")


def test_apply_quality_filters_function(books: List[Dict[str, Any]], sampling_policy: Dict[str, Any]):
    """Test the complete apply_quality_filters function."""
    logger = logging.getLogger(__name__)
    
    logger.info("=== TESTING COMPLETE APPLY_QUALITY_FILTERS FUNCTION ===")
    
    from data_processing.quality_filters import apply_quality_filters
    
    try:
        final_books = apply_quality_filters(books, sampling_policy)
        logger.info(f"Final books from apply_quality_filters: {len(final_books)}")
        
        if len(final_books) == 0:
            logger.error("apply_quality_filters returned 0 books!")
        else:
            logger.info("apply_quality_filters working correctly!")
            
    except Exception as e:
        logger.error(f"apply_quality_filters failed: {e}", exc_info=True)


def main():
    """Main test function."""
    logger = setup_logging()
    
    logger.info("Starting author balancing test...")
    
    try:
        # Load configuration
        config_loader = ConfigLoader("config")
        sampling_policy = config_loader.get_sampling_policy()
        
        # Load sample data
        books = load_and_convert_sample_data(1000)
        
        # Analyze author data structure
        analyze_author_data_structure(books)
        
        # Test step by step
        test_author_balancing_step_by_step(books, sampling_policy)
        
        # Test complete function
        test_apply_quality_filters_function(books, sampling_policy)
        
        logger.info("Test completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
