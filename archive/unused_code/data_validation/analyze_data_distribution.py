#!/usr/bin/env python3
"""
Analyze data distribution to understand why quality filters are too strict.
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
            logging.FileHandler(f'logs/validation/data_distribution_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger(__name__)


def load_sample_data(sample_size: int = 10000) -> List[Dict[str, Any]]:
    """Load sample books data for analysis."""
    logger = logging.getLogger(__name__)
    
    data_path = Path('data/raw/goodreads_books_romance.json.gz')
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading {sample_size} sample books for analysis...")
    
    with gzip.open(data_path, 'rt') as f:
        books = []
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            books.append(json.loads(line))
    
    logger.info(f"Loaded {len(books)} books for analysis")
    return books


def analyze_ratings_distribution(books: List[Dict[str, Any]]):
    """Analyze the distribution of ratings counts."""
    logger = logging.getLogger(__name__)
    
    logger.info("=== RATINGS COUNT DISTRIBUTION ===")
    
    ratings_counts = []
    for book in books:
        try:
            rating_count = int(book.get('ratings_count', 0))
            ratings_counts.append(rating_count)
        except (ValueError, TypeError):
            ratings_counts.append(0)
    
    df = pd.DataFrame({'ratings_count': ratings_counts})
    
    logger.info(f"Total books analyzed: {len(df)}")
    logger.info(f"Books with 0 ratings: {len(df[df['ratings_count'] == 0])}")
    logger.info(f"Books with 1-5 ratings: {len(df[(df['ratings_count'] >= 1) & (df['ratings_count'] <= 5)])}")
    logger.info(f"Books with 6-10 ratings: {len(df[(df['ratings_count'] >= 6) & (df['ratings_count'] <= 10)])}")
    logger.info(f"Books with 11-50 ratings: {len(df[(df['ratings_count'] >= 11) & (df['ratings_count'] <= 50)])}")
    logger.info(f"Books with 51-100 ratings: {len(df[(df['ratings_count'] >= 51) & (df['ratings_count'] <= 100)])}")
    logger.info(f"Books with 101+ ratings: {len(df[df['ratings_count'] >= 101])}")
    
    logger.info(f"Percentiles:")
    logger.info(f"  25th percentile: {df['ratings_count'].quantile(0.25):.0f}")
    logger.info(f"  50th percentile (median): {df['ratings_count'].quantile(0.50):.0f}")
    logger.info(f"  75th percentile: {df['ratings_count'].quantile(0.75):.0f}")
    logger.info(f"  90th percentile: {df['ratings_count'].quantile(0.90):.0f}")
    logger.info(f"  95th percentile: {df['ratings_count'].quantile(0.95):.0f}")
    
    # Test different thresholds
    thresholds = [1, 5, 10, 20, 50, 100]
    for threshold in thresholds:
        passing = len(df[df['ratings_count'] >= threshold])
        percentage = (passing / len(df)) * 100
        logger.info(f"  Books with ≥{threshold} ratings: {passing} ({percentage:.1f}%)")


def analyze_publication_year_distribution(books: List[Dict[str, Any]]):
    """Analyze the distribution of publication years."""
    logger = logging.getLogger(__name__)
    
    logger.info("=== PUBLICATION YEAR DISTRIBUTION ===")
    
    years = []
    for book in books:
        try:
            year_str = book.get('publication_year', '')
            if year_str and year_str.strip():
                year = int(year_str)
                years.append(year)
        except (ValueError, TypeError):
            continue
    
    if not years:
        logger.warning("No valid publication years found!")
        return
    
    df = pd.DataFrame({'publication_year': years})
    
    logger.info(f"Books with valid publication years: {len(df)}")
    logger.info(f"Books with missing/invalid years: {len(books) - len(df)}")
    
    logger.info(f"Year range: {df['publication_year'].min()} - {df['publication_year'].max()}")
    
    # Analyze by decades
    decades = {
        '1800s': len(df[df['publication_year'] < 1900]),
        '1900s': len(df[(df['publication_year'] >= 1900) & (df['publication_year'] < 2000)]),
        '2000s': len(df[(df['publication_year'] >= 2000) & (df['publication_year'] < 2010)]),
        '2010s': len(df[(df['publication_year'] >= 2010) & (df['publication_year'] < 2020)]),
        '2020s': len(df[df['publication_year'] >= 2020])
    }
    
    for decade, count in decades.items():
        percentage = (count / len(df)) * 100
        logger.info(f"  {decade}: {count} books ({percentage:.1f}%)")
    
    # Test different year thresholds
    thresholds = [1900, 1950, 1980, 1990, 2000, 2005, 2010, 2015]
    for threshold in thresholds:
        passing = len(df[df['publication_year'] >= threshold])
        percentage = (passing / len(df)) * 100
        logger.info(f"  Books from ≥{threshold}: {passing} ({percentage:.1f}%)")


def analyze_language_distribution(books: List[Dict[str, Any]]):
    """Analyze the distribution of languages."""
    logger = logging.getLogger(__name__)
    
    logger.info("=== LANGUAGE DISTRIBUTION ===")
    
    languages = {}
    for book in books:
        lang = book.get('language_code', '')
        if lang:
            languages[lang] = languages.get(lang, 0) + 1
        else:
            languages['empty'] = languages.get('empty', 0) + 1
    
    total = len(books)
    logger.info(f"Total books: {total}")
    
    # Sort by count
    sorted_langs = sorted(languages.items(), key=lambda x: x[1], reverse=True)
    
    for lang, count in sorted_langs[:20]:  # Top 20 languages
        percentage = (count / total) * 100
        logger.info(f"  {lang}: {count} books ({percentage:.1f}%)")
    
    # Check English variants
    english_variants = ['eng', 'en-US', 'en-GB', 'en-CA', 'en-AU', 'en-NZ', '']
    english_count = sum(languages.get(lang, 0) for lang in english_variants)
    english_percentage = (english_count / total) * 100
    logger.info(f"  Total English books: {english_count} ({english_percentage:.1f}%)")


def analyze_description_distribution(books: List[Dict[str, Any]]):
    """Analyze the distribution of descriptions."""
    logger = logging.getLogger(__name__)
    
    logger.info("=== DESCRIPTION DISTRIBUTION ===")
    
    descriptions = []
    for book in books:
        desc = book.get('description', '')
        if desc and desc.strip():
            descriptions.append(len(desc.strip()))
        else:
            descriptions.append(0)
    
    df = pd.DataFrame({'description_length': descriptions})
    
    logger.info(f"Books with descriptions: {len(df[df['description_length'] > 0])}")
    logger.info(f"Books without descriptions: {len(df[df['description_length'] == 0])}")
    
    if len(df[df['description_length'] > 0]) > 0:
        logger.info(f"Description length statistics (for books with descriptions):")
        logger.info(f"  Mean: {df[df['description_length'] > 0]['description_length'].mean():.0f} characters")
        logger.info(f"  Median: {df[df['description_length'] > 0]['description_length'].median():.0f} characters")
        logger.info(f"  Min: {df[df['description_length'] > 0]['description_length'].min():.0f} characters")
        logger.info(f"  Max: {df[df['description_length'] > 0]['description_length'].max():.0f} characters")


def suggest_filter_adjustments(books: List[Dict[str, Any]]):
    """Suggest filter adjustments based on data analysis."""
    logger = logging.getLogger(__name__)
    
    logger.info("=== FILTER ADJUSTMENT SUGGESTIONS ===")
    
    # Analyze current filter impact
    current_filters = {
        'ratings_count_min': 10,
        'publication_year_min': 2000,
        'language_code': 'eng'
    }
    
    passing_books = 0
    total_books = len(books)
    
    for book in books:
        # Check ratings
        try:
            rating_count = int(book.get('ratings_count', 0))
            if rating_count < current_filters['ratings_count_min']:
                continue
        except (ValueError, TypeError):
            continue
        
        # Check publication year
        try:
            year_str = book.get('publication_year', '')
            if year_str and year_str.strip():
                year = int(year_str)
                if year < current_filters['publication_year_min']:
                    continue
            else:
                continue
        except (ValueError, TypeError):
            continue
        
        # Check language
        lang = book.get('language_code', '')
        english_variants = ['eng', 'en-US', 'en-GB', 'en-CA', 'en-AU', 'en-NZ', '']
        if lang not in english_variants:
            continue
        
        # Check description
        desc = book.get('description', '')
        if not desc or not desc.strip():
            continue
        
        passing_books += 1
    
    current_percentage = (passing_books / total_books) * 100
    logger.info(f"Current filters pass: {passing_books}/{total_books} books ({current_percentage:.1f}%)")
    
    # Suggest adjustments
    logger.info("Suggested filter adjustments:")
    
    # Ratings count suggestions
    ratings_thresholds = [1, 5, 10, 20, 50]
    for threshold in ratings_thresholds:
        passing = 0
        for book in books:
            try:
                rating_count = int(book.get('ratings_count', 0))
                if rating_count >= threshold:
                    passing += 1
            except (ValueError, TypeError):
                continue
        percentage = (passing / total_books) * 100
        logger.info(f"  Ratings ≥{threshold}: {passing} books ({percentage:.1f}%)")
    
    # Publication year suggestions
    year_thresholds = [1980, 1990, 2000, 2005, 2010]
    for threshold in year_thresholds:
        passing = 0
        for book in books:
            try:
                year_str = book.get('publication_year', '')
                if year_str and year_str.strip():
                    year = int(year_str)
                    if year >= threshold:
                        passing += 1
            except (ValueError, TypeError):
                continue
        percentage = (passing / total_books) * 100
        logger.info(f"  Year ≥{threshold}: {passing} books ({percentage:.1f}%)")


def main():
    """Main analysis function."""
    logger = setup_logging()
    
    logger.info("Starting data distribution analysis...")
    
    try:
        # Load sample data
        books = load_sample_data(10000)
        
        # Analyze distributions
        analyze_ratings_distribution(books)
        analyze_publication_year_distribution(books)
        analyze_language_distribution(books)
        analyze_description_distribution(books)
        
        # Suggest adjustments
        suggest_filter_adjustments(books)
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
