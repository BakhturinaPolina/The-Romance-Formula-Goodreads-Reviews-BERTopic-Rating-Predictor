#!/usr/bin/env python3
"""
Enhanced Quality Filter Debugging Script
Diagnoses why quality filters are returning 0 books and identifies field mapping issues.
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

from data_processing.quality_filters import QualityFilters, apply_quality_filters
from data_processing.config_loader import ConfigLoader


def setup_logging():
    """Set up logging for debugging."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'logs/validation/filter_debugging_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger(__name__)


def load_sample_data(sample_size: int = 1000) -> List[Dict[str, Any]]:
    """Load sample books data for testing."""
    logger = logging.getLogger(__name__)
    
    data_path = Path('data/raw/goodreads_books_romance.json.gz')
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading {sample_size} sample books for debugging...")
    
    with gzip.open(data_path, 'rt') as f:
        books = []
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            books.append(json.loads(line))
    
    logger.info(f"Loaded {len(books)} books for testing")
    return books


def analyze_configuration():
    """Analyze if configuration is loading correctly."""
    logger = logging.getLogger(__name__)
    
    logger.info("=== CONFIGURATION ANALYSIS ===")
    
    try:
        config_loader = ConfigLoader("config")
        sampling_policy = config_loader.get_sampling_policy()
        variable_selection = config_loader.get_variable_selection()
        
        logger.info(f"Sampling policy loaded successfully")
        logger.info(f"Variable selection keys: {list(variable_selection.keys())}")
        
        # Check quality filters configuration
        quality_filters = sampling_policy.get('quality_filters', {})
        logger.info(f"Quality filters config: {quality_filters}")
        
        return sampling_policy, variable_selection
        
    except Exception as e:
        logger.error(f"Configuration loading failed: {str(e)}")
        raise


def analyze_individual_filters(books: List[Dict[str, Any]], sampling_policy: Dict[str, Any]):
    """Analyze each filter individually to see where books are being rejected."""
    logger = logging.getLogger(__name__)
    
    logger.info("=== INDIVIDUAL FILTER ANALYSIS ===")
    
    filters = QualityFilters(sampling_policy)
    
    # Get filter configuration
    filter_config = filters.get_filter_summary()
    logger.info(f"Filter configuration: {filter_config}")
    
    # Track statistics for each filter
    filter_stats = {
        'total_books': len(books),
        'publication_year_pass': 0,
        'language_pass': 0,
        'ratings_count_pass': 0,
        'description_pass': 0,
        'all_filters_pass': 0
    }
    
    sample_failing_books = {
        'publication_year': [],
        'language': [],
        'ratings_count': [],
        'description': []
    }
    
    for i, book in enumerate(books):
        # Check each filter individually
        year_pass = filters._check_publication_year(book)
        lang_pass = filters._check_language(book)
        ratings_pass = filters._check_ratings_count(book)
        desc_pass = filters._check_description_completeness(book)
        
        if year_pass:
            filter_stats['publication_year_pass'] += 1
        else:
            if len(sample_failing_books['publication_year']) < 5:
                sample_failing_books['publication_year'].append({
                    'title': book.get('title', 'N/A'),
                    'publication_year': book.get('publication_year'),
                    'book_id': book.get('book_id')
                })
        
        if lang_pass:
            filter_stats['language_pass'] += 1
        else:
            if len(sample_failing_books['language']) < 5:
                sample_failing_books['language'].append({
                    'title': book.get('title', 'N/A'),
                    'language_code': book.get('language_code'),
                    'book_id': book.get('book_id')
                })
        
        if ratings_pass:
            filter_stats['ratings_count_pass'] += 1
        else:
            if len(sample_failing_books['ratings_count']) < 5:
                sample_failing_books['ratings_count'].append({
                    'title': book.get('title', 'N/A'),
                    'ratings_count': book.get('ratings_count'),
                    'book_id': book.get('book_id')
                })
        
        if desc_pass:
            filter_stats['description_pass'] += 1
        else:
            if len(sample_failing_books['description']) < 5:
                desc = book.get('description', '')
                sample_failing_books['description'].append({
                    'title': book.get('title', 'N/A'),
                    'description_length': len(desc),
                    'has_description': bool(desc and desc.strip()),
                    'book_id': book.get('book_id')
                })
        
        if year_pass and lang_pass and ratings_pass and desc_pass:
            filter_stats['all_filters_pass'] += 1
    
    # Log statistics
    logger.info(f"Filter statistics: {filter_stats}")
    
    # Log sample failing books
    for filter_name, failing_books in sample_failing_books.items():
        if failing_books:
            logger.info(f"Sample books failing {filter_name} filter:")
            for book in failing_books:
                logger.info(f"  {book}")
    
    return filter_stats, sample_failing_books


def analyze_data_types_and_fields(books: List[Dict[str, Any]]):
    """Analyze the data types and field structure of the books."""
    logger = logging.getLogger(__name__)
    
    logger.info("=== DATA STRUCTURE ANALYSIS ===")
    
    if not books:
        logger.error("No books provided for analysis")
        return
    
    sample_book = books[0]
    logger.info(f"Sample book keys: {list(sample_book.keys())}")
    
    # Analyze critical fields
    critical_fields = ['publication_year', 'language_code', 'ratings_count', 'description', 'book_id']
    
    for field in critical_fields:
        values = []
        types = []
        missing_count = 0
        
        for book in books[:100]:  # Sample first 100 books
            value = book.get(field)
            if value is None or value == '':
                missing_count += 1
            else:
                values.append(value)
                types.append(type(value).__name__)
        
        unique_types = list(set(types))
        sample_values = values[:10] if values else []
        
        logger.info(f"Field '{field}':")
        logger.info(f"  Types found: {unique_types}")
        logger.info(f"  Missing/empty count: {missing_count}/100")
        logger.info(f"  Sample values: {sample_values}")


def test_pipeline_vs_individual(books: List[Dict[str, Any]], sampling_policy: Dict[str, Any]):
    """Test the difference between individual QualityFilters and the full pipeline."""
    logger = logging.getLogger(__name__)
    
    logger.info("=== PIPELINE VS INDIVIDUAL COMPARISON ===")
    
    # Test individual QualityFilters
    quality_filters = QualityFilters(sampling_policy)
    individual_result = quality_filters.filter_books(books)
    
    logger.info(f"Individual QualityFilters result: {len(individual_result)} books passed")
    
    # Test full pipeline
    try:
        full_pipeline_result = apply_quality_filters(books, sampling_policy)
        logger.info(f"Full pipeline result: {len(full_pipeline_result)} books passed")
    except Exception as e:
        logger.error(f"Full pipeline failed: {str(e)}")
        full_pipeline_result = []
    
    return individual_result, full_pipeline_result


def main():
    """Main debugging function."""
    logger = setup_logging()
    
    logger.info("Starting enhanced quality filter debugging...")
    
    try:
        # Load configuration
        sampling_policy, variable_selection = analyze_configuration()
        
        # Load sample data
        books = load_sample_data(1000)
        
        # Analyze data structure
        analyze_data_types_and_fields(books)
        
        # Analyze individual filters
        filter_stats, failing_books = analyze_individual_filters(books, sampling_policy)
        
        # Test pipeline vs individual
        individual_result, pipeline_result = test_pipeline_vs_individual(books, sampling_policy)
        
        # Summary
        logger.info("=== DEBUGGING SUMMARY ===")
        logger.info(f"Total books tested: {len(books)}")
        logger.info(f"Individual QualityFilters passed: {len(individual_result)}")
        logger.info(f"Full pipeline passed: {len(pipeline_result)}")
        logger.info(f"Filter statistics: {filter_stats}")
        
        # Save results to file
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_books_tested': len(books),
            'individual_filters_passed': len(individual_result),
            'full_pipeline_passed': len(pipeline_result),
            'filter_statistics': filter_stats,
            'sample_failing_books': failing_books,
            'configuration': {
                'sampling_policy': sampling_policy,
                'variable_selection_keys': list(variable_selection.keys())
            }
        }
        
        results_path = f'logs/validation/filter_debugging_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {results_path}")
        
    except Exception as e:
        logger.error(f"Debugging failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
