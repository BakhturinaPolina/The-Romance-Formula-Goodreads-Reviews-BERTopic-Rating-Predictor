#!/usr/bin/env python3
"""
Step-by-Step Filter Analysis
Applies quality filters one by one with EDA statistics after each step.
"""

import gzip
import json
import sys
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_processing.config_loader import ConfigLoader
from data_processing.quality_filters import QualityFilters


def setup_logging():
    """Set up logging for analysis."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'logs/validation/step_by_step_filter_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger(__name__)


def load_and_convert_data_types(sample_size: int = 10000) -> List[Dict[str, Any]]:
    """Load sample data and convert data types properly."""
    logger = logging.getLogger(__name__)
    
    logger.info("=== STEP 0: LOADING AND CONVERTING DATA TYPES ===")
    
    data_path = Path('data/raw/goodreads_books_romance.json.gz')
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading {sample_size} sample books...")
    
    # Load raw data
    with gzip.open(data_path, 'rt') as f:
        raw_books = []
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            raw_books.append(json.loads(line))
    
    logger.info(f"Loaded {len(raw_books)} raw books")
    
    # Convert data types
    logger.info("Converting data types...")
    converted_books = []
    
    # Define conversion mappings
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
    
    conversion_errors = 0
    
    for book in raw_books:
        converted_book = book.copy()
        
        # Apply numeric conversions
        for field, conversion_func in numeric_conversions.items():
            if field in converted_book:
                try:
                    value = converted_book[field]
                    if value is not None and value != '':
                        converted_book[field] = conversion_func(value)
                except (ValueError, TypeError):
                    conversion_errors += 1
                    # Keep original value if conversion fails
                    pass
        
        converted_books.append(converted_book)
    
    logger.info(f"Data type conversion completed with {conversion_errors} errors")
    
    # Verify data types
    logger.info("Verifying data types...")
    sample_book = converted_books[0]
    for field, expected_type in numeric_conversions.items():
        if field in sample_book:
            actual_type = type(sample_book[field]).__name__
            logger.info(f"  {field}: {actual_type} (expected: {expected_type.__name__})")
    
    return converted_books


def analyze_data_after_step(books: List[Dict[str, Any]], step_name: str, step_number: int):
    """Analyze data characteristics after each filter step."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"=== STEP {step_number}: {step_name.upper()} ===")
    logger.info(f"Books remaining: {len(books)}")
    
    if len(books) == 0:
        logger.warning("⚠️  No books remaining after this step!")
        return
    
    # Analyze ratings distribution
    ratings_counts = []
    for book in books:
        try:
            rating_count = book.get('ratings_count', 0)
            if isinstance(rating_count, (int, float)):
                ratings_counts.append(int(rating_count))
            else:
                ratings_counts.append(0)
        except (ValueError, TypeError):
            ratings_counts.append(0)
    
    df_ratings = pd.DataFrame({'ratings_count': ratings_counts})
    
    logger.info(f"Ratings count statistics:")
    logger.info(f"  Mean: {df_ratings['ratings_count'].mean():.1f}")
    logger.info(f"  Median: {df_ratings['ratings_count'].median():.1f}")
    logger.info(f"  Min: {df_ratings['ratings_count'].min()}")
    logger.info(f"  Max: {df_ratings['ratings_count'].max()}")
    logger.info(f"  Books with 0 ratings: {len(df_ratings[df_ratings['ratings_count'] == 0])}")
    logger.info(f"  Books with 1-5 ratings: {len(df_ratings[(df_ratings['ratings_count'] >= 1) & (df_ratings['ratings_count'] <= 5)])}")
    logger.info(f"  Books with 6-10 ratings: {len(df_ratings[(df_ratings['ratings_count'] >= 6) & (df_ratings['ratings_count'] <= 10)])}")
    logger.info(f"  Books with 11+ ratings: {len(df_ratings[df_ratings['ratings_count'] >= 11])}")
    
    # Analyze publication years
    years = []
    for book in books:
        try:
            year = book.get('publication_year')
            if isinstance(year, (int, float)) and year > 0:
                years.append(int(year))
        except (ValueError, TypeError):
            continue
    
    if years:
        df_years = pd.DataFrame({'publication_year': years})
        logger.info(f"Publication year statistics:")
        logger.info(f"  Mean: {df_years['publication_year'].mean():.1f}")
        logger.info(f"  Median: {df_years['publication_year'].median():.1f}")
        logger.info(f"  Min: {df_years['publication_year'].min()}")
        logger.info(f"  Max: {df_years['publication_year'].max()}")
        
        # Decades breakdown
        decades = {
            '1800s': len(df_years[df_years['publication_year'] < 1900]),
            '1900s': len(df_years[(df_years['publication_year'] >= 1900) & (df_years['publication_year'] < 2000)]),
            '2000s': len(df_years[(df_years['publication_year'] >= 2000) & (df_years['publication_year'] < 2010)]),
            '2010s': len(df_years[(df_years['publication_year'] >= 2010) & (df_years['publication_year'] < 2020)]),
            '2020s': len(df_years[df_years['publication_year'] >= 2020])
        }
        
        for decade, count in decades.items():
            if count > 0:
                percentage = (count / len(df_years)) * 100
                logger.info(f"  {decade}: {count} books ({percentage:.1f}%)")
    else:
        logger.warning("No valid publication years found")
    
    # Analyze languages
    languages = {}
    for book in books:
        lang = book.get('language_code', '')
        if lang:
            languages[lang] = languages.get(lang, 0) + 1
        else:
            languages['empty'] = languages.get('empty', 0) + 1
    
    logger.info(f"Language distribution (top 10):")
    sorted_langs = sorted(languages.items(), key=lambda x: x[1], reverse=True)
    for lang, count in sorted_langs[:10]:
        percentage = (count / len(books)) * 100
        logger.info(f"  {lang}: {count} books ({percentage:.1f}%)")
    
    # Analyze descriptions
    descriptions = []
    for book in books:
        desc = book.get('description', '')
        if desc and desc.strip():
            descriptions.append(len(desc.strip()))
        else:
            descriptions.append(0)
    
    df_descriptions = pd.DataFrame({'description_length': descriptions})
    books_with_desc = len(df_descriptions[df_descriptions['description_length'] > 0])
    books_without_desc = len(df_descriptions[df_descriptions['description_length'] == 0])
    
    logger.info(f"Description statistics:")
    logger.info(f"  Books with descriptions: {books_with_desc}")
    logger.info(f"  Books without descriptions: {books_without_desc}")
    if books_with_desc > 0:
        logger.info(f"  Mean description length: {df_descriptions[df_descriptions['description_length'] > 0]['description_length'].mean():.0f} characters")
        logger.info(f"  Median description length: {df_descriptions[df_descriptions['description_length'] > 0]['description_length'].median():.0f} characters")


def apply_filters_step_by_step(books: List[Dict[str, Any]], sampling_policy: Dict[str, Any]):
    """Apply filters one by one with analysis after each step."""
    logger = logging.getLogger(__name__)
    
    logger.info("=== APPLYING FILTERS STEP BY STEP ===")
    
    # Get filter configuration
    quality_filters = sampling_policy.get('quality_filters', {})
    logger.info(f"Filter configuration: {quality_filters}")
    
    # Initialize with all books
    current_books = books.copy()
    
    # Step 1: Ratings count filter
    logger.info("\n" + "="*50)
    ratings_min = quality_filters.get('ratings_count_min', 10)
    logger.info(f"Applying ratings count filter (≥{ratings_min} ratings)...")
    
    filtered_books = []
    for book in current_books:
        try:
            rating_count = book.get('ratings_count', 0)
            if isinstance(rating_count, (int, float)) and rating_count >= ratings_min:
                filtered_books.append(book)
        except (ValueError, TypeError):
            continue
    
    current_books = filtered_books
    analyze_data_after_step(current_books, "ratings count filter", 1)
    
    # Step 2: Publication year filter
    logger.info("\n" + "="*50)
    year_min = quality_filters.get('publication_year_min', 2000)
    logger.info(f"Applying publication year filter (≥{year_min})...")
    
    filtered_books = []
    for book in current_books:
        try:
            year = book.get('publication_year')
            if isinstance(year, (int, float)) and year >= year_min:
                filtered_books.append(book)
        except (ValueError, TypeError):
            continue
    
    current_books = filtered_books
    analyze_data_after_step(current_books, "publication year filter", 2)
    
    # Step 3: Language filter
    logger.info("\n" + "="*50)
    required_language = quality_filters.get('language_code', 'eng')
    logger.info(f"Applying language filter (English variants)...")
    
    english_variants = ['eng', 'en-US', 'en-GB', 'en-CA', 'en-AU', 'en-NZ', '']
    filtered_books = []
    for book in current_books:
        lang = book.get('language_code', '')
        if lang in english_variants:
            filtered_books.append(book)
    
    current_books = filtered_books
    analyze_data_after_step(current_books, "language filter", 3)
    
    # Step 4: Description filter
    logger.info("\n" + "="*50)
    logger.info("Applying description completeness filter...")
    
    filtered_books = []
    for book in current_books:
        desc = book.get('description', '')
        if desc and desc.strip():
            filtered_books.append(book)
    
    current_books = filtered_books
    analyze_data_after_step(current_books, "description filter", 4)
    
    # Final summary
    logger.info("\n" + "="*50)
    logger.info("=== FINAL SUMMARY ===")
    logger.info(f"Original books: {len(books)}")
    logger.info(f"Final books after all filters: {len(current_books)}")
    logger.info(f"Retention rate: {(len(current_books) / len(books)) * 100:.1f}%")
    
    return current_books


def test_different_filter_combinations(books: List[Dict[str, Any]], sampling_policy: Dict[str, Any]):
    """Test different filter combinations to find optimal settings."""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*50)
    logger.info("=== TESTING DIFFERENT FILTER COMBINATIONS ===")
    
    # Test different ratings thresholds
    ratings_thresholds = [1, 5, 10, 20, 50]
    year_thresholds = [1980, 1990, 2000, 2005, 2010]
    
    results = []
    
    for ratings_min in ratings_thresholds:
        for year_min in year_thresholds:
            # Apply filters
            filtered_books = []
            for book in books:
                # Ratings filter
                try:
                    rating_count = book.get('ratings_count', 0)
                    if not isinstance(rating_count, (int, float)) or rating_count < ratings_min:
                        continue
                except (ValueError, TypeError):
                    continue
                
                # Year filter
                try:
                    year = book.get('publication_year')
                    if not isinstance(year, (int, float)) or year < year_min:
                        continue
                except (ValueError, TypeError):
                    continue
                
                # Language filter
                lang = book.get('language_code', '')
                english_variants = ['eng', 'en-US', 'en-GB', 'en-CA', 'en-AU', 'en-NZ', '']
                if lang not in english_variants:
                    continue
                
                # Description filter
                desc = book.get('description', '')
                if not desc or not desc.strip():
                    continue
                
                filtered_books.append(book)
            
            retention_rate = (len(filtered_books) / len(books)) * 100
            results.append({
                'ratings_min': ratings_min,
                'year_min': year_min,
                'books_passing': len(filtered_books),
                'retention_rate': retention_rate
            })
    
    # Sort by retention rate (descending)
    results.sort(key=lambda x: x['retention_rate'], reverse=True)
    
    logger.info("Top 10 filter combinations by retention rate:")
    for i, result in enumerate(results[:10]):
        logger.info(f"  {i+1}. Ratings≥{result['ratings_min']}, Year≥{result['year_min']}: "
                   f"{result['books_passing']} books ({result['retention_rate']:.1f}%)")


def main():
    """Main analysis function."""
    logger = setup_logging()
    
    logger.info("Starting step-by-step filter analysis...")
    
    try:
        # Load configuration
        config_loader = ConfigLoader("config")
        sampling_policy = config_loader.get_sampling_policy()
        
        # Step 0: Load and convert data types
        books = load_and_convert_data_types(10000)
        
        # Step 1-4: Apply filters step by step
        final_books = apply_filters_step_by_step(books, sampling_policy)
        
        # Test different combinations
        test_different_filter_combinations(books, sampling_policy)
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
