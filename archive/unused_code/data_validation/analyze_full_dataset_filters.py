#!/usr/bin/env python3
"""
Analyze Full Dataset Filter Characteristics
Find optimal filter settings for the full dataset.
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


def setup_logging():
    """Set up logging for analysis."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'logs/validation/analyze_full_dataset_filters_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger(__name__)


def load_full_dataset_sample(sample_size: int = 50000) -> List[Dict[str, Any]]:
    """Load a larger sample from the full dataset."""
    logger = logging.getLogger(__name__)
    
    data_path = Path('data/raw/goodreads_books_romance.json.gz')
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading {sample_size} sample books from full dataset...")
    
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


def analyze_dataset_characteristics(books: List[Dict[str, Any]]):
    """Analyze the characteristics of the full dataset."""
    logger = logging.getLogger(__name__)
    
    logger.info("=== ANALYZING FULL DATASET CHARACTERISTICS ===")
    
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
    
    # Percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = df_ratings['ratings_count'].quantile(p/100)
        logger.info(f"  {p}th percentile: {value:.0f}")
    
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
    
    # Analyze languages
    languages = {}
    for book in books:
        lang = book.get('language_code', '')
        if lang:
            languages[lang] = languages.get(lang, 0) + 1
        else:
            languages['empty'] = languages.get('empty', 0) + 1
    
    logger.info(f"Language distribution (top 15):")
    sorted_langs = sorted(languages.items(), key=lambda x: x[1], reverse=True)
    for lang, count in sorted_langs[:15]:
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


def test_filter_combinations(books: List[Dict[str, Any]]):
    """Test different filter combinations to find optimal settings."""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*50)
    logger.info("=== TESTING FILTER COMBINATIONS ===")
    
    # Test different ratings thresholds
    ratings_thresholds = [0, 1, 2, 5, 10, 20, 50, 100]
    year_thresholds = [1900, 1950, 1980, 1990, 2000, 2005, 2010]
    
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
                
                # Language filter (English variants)
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
    
    logger.info("Top 15 filter combinations by retention rate:")
    for i, result in enumerate(results[:15]):
        logger.info(f"  {i+1}. Ratings≥{result['ratings_min']}, Year≥{result['year_min']}: "
                   f"{result['books_passing']} books ({result['retention_rate']:.1f}%)")
    
    # Find combinations that give reasonable dataset sizes
    logger.info("\nFilter combinations for different dataset sizes:")
    target_sizes = [1000, 5000, 10000, 20000, 50000]
    
    for target_size in target_sizes:
        # Find the most lenient combination that gives at least target_size books
        for result in results:
            if result['books_passing'] >= target_size:
                logger.info(f"  For {target_size:,} books: Ratings≥{result['ratings_min']}, Year≥{result['year_min']} "
                           f"({result['books_passing']} books, {result['retention_rate']:.1f}%)")
                break


def recommend_optimal_filters(books: List[Dict[str, Any]]):
    """Recommend optimal filter settings based on analysis."""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*50)
    logger.info("=== RECOMMENDED OPTIMAL FILTERS ===")
    
    # Analyze current data to recommend settings
    ratings_counts = []
    years = []
    
    for book in books:
        try:
            rating_count = book.get('ratings_count', 0)
            if isinstance(rating_count, (int, float)):
                ratings_counts.append(int(rating_count))
        except (ValueError, TypeError):
            ratings_counts.append(0)
        
        try:
            year = book.get('publication_year')
            if isinstance(year, (int, float)) and year > 0:
                years.append(int(year))
        except (ValueError, TypeError):
            continue
    
    if ratings_counts and years:
        df_ratings = pd.DataFrame({'ratings_count': ratings_counts})
        df_years = pd.DataFrame({'publication_year': years})
        
        # Recommend ratings threshold that keeps 80% of books
        ratings_80th = df_ratings['ratings_count'].quantile(0.2)  # 20th percentile = 80% keep
        ratings_90th = df_ratings['ratings_count'].quantile(0.1)  # 10th percentile = 90% keep
        
        # Recommend year threshold that keeps 80% of books
        year_80th = df_years['publication_year'].quantile(0.2)  # 20th percentile = 80% keep
        year_90th = df_years['publication_year'].quantile(0.1)  # 10th percentile = 90% keep
        
        logger.info("Recommended filter settings:")
        logger.info(f"  Conservative (80% retention):")
        logger.info(f"    ratings_count_min: {max(1, int(ratings_80th))}")
        logger.info(f"    publication_year_min: {max(1900, int(year_80th))}")
        logger.info(f"  Liberal (90% retention):")
        logger.info(f"    ratings_count_min: {max(1, int(ratings_90th))}")
        logger.info(f"    publication_year_min: {max(1900, int(year_90th))}")


def main():
    """Main analysis function."""
    logger = setup_logging()
    
    logger.info("Starting full dataset filter analysis...")
    
    try:
        # Load larger sample from full dataset
        books = load_full_dataset_sample(50000)
        
        # Analyze characteristics
        analyze_dataset_characteristics(books)
        
        # Test filter combinations
        test_filter_combinations(books)
        
        # Recommend optimal filters
        recommend_optimal_filters(books)
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
