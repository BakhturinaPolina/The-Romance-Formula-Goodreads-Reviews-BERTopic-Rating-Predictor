#!/usr/bin/env python3
"""
Extract English reviews for romance books from Goodreads reviews dataset.

This script:
1. Reads book IDs from romance_books_main_final.csv
2. Filters reviews from goodreads_reviews_romance.json.gz matching those book IDs
3. Detects and keeps only English reviews using language detection
4. Outputs a single CSV with columns: review_id, review_text, rating, book_id
"""

import ast
import csv
import gzip
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Set

import pandas as pd
from langdetect import detect, LangDetectException

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
BOOKS_CSV = Path("/home/polina/Documents/goodreads_romance_research_cursor/anna_archive_romance_pipeline/data/processed/romance_books_main_final.csv")
REVIEWS_GZ = PROJECT_ROOT / "data" / "raw" / "goodreads_reviews_romance.json.gz"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "romance_reviews_english.csv"

# Constants
MIN_REVIEW_LENGTH = 10  # Skip reviews shorter than this
PROGRESS_INTERVAL = 5000  # Log progress every N reviews (more frequent for monitoring)


def setup_logging(log_dir: Path) -> logging.Logger:
    """
    Set up logging to both file and console.
    
    Args:
        log_dir: Directory to store log files
        
    Returns:
        Configured logger instance
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"extract_reviews_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


def load_book_ids(csv_path: Path, logger: logging.Logger) -> Set[str]:
    """
    Load book IDs from the CSV file.
    
    Args:
        csv_path: Path to the romance books CSV file
        logger: Logger instance for logging
        
    Returns:
        Set of book IDs as strings
    """
    logger.info(f"Loading book IDs from {csv_path}...")
    start_time = time.time()
    
    df = pd.read_csv(csv_path)
    logger.info(f"CSV loaded: {len(df)} rows, {len(df.columns)} columns")
    
    if 'book_id_list_en' not in df.columns:
        raise ValueError(f"Column 'book_id_list_en' not found in CSV. Available columns: {df.columns.tolist()}")
    
    book_ids = set()
    parse_errors = 0
    
    for idx, row in df.iterrows():
        book_id_list_str = row['book_id_list_en']
        
        # Skip NaN or empty values
        if pd.isna(book_id_list_str) or not book_id_list_str or book_id_list_str.strip() == '':
            continue
        
        try:
            # Parse string representation of list to actual list
            book_id_list = ast.literal_eval(book_id_list_str)
            
            # Add all book IDs from this row to the set
            for book_id in book_id_list:
                book_ids.add(str(book_id))
                
        except (ValueError, SyntaxError) as e:
            parse_errors += 1
            if parse_errors <= 10:  # Only log first 10 errors
                logger.warning(f"Could not parse book_id_list_en at row {idx}: {book_id_list_str[:50]}... Error: {e}")
            continue
    
    elapsed = time.time() - start_time
    logger.info(f"Loaded {len(book_ids):,} unique book IDs in {elapsed:.2f} seconds")
    if parse_errors > 0:
        logger.warning(f"Encountered {parse_errors} parse errors (only first 10 logged)")
    
    return book_ids


def is_english_review(review_text: str) -> bool:
    """
    Check if a review is in English using language detection.
    
    Args:
        review_text: The review text to check
        
    Returns:
        True if the review is detected as English, False otherwise
    """
    if not review_text or len(review_text.strip()) < MIN_REVIEW_LENGTH:
        return False
    
    try:
        detected_lang = detect(review_text)
        return detected_lang == 'en'
    except LangDetectException:
        # If detection fails, skip the review
        return False


def extract_reviews(book_ids: Set[str], reviews_gz: Path, output_csv: Path, logger: logging.Logger):
    """
    Extract English reviews for the specified book IDs.
    
    Args:
        book_ids: Set of book IDs to filter by
        reviews_gz: Path to the gzipped reviews JSON file
        output_csv: Path to output CSV file
        logger: Logger instance for logging
    """
    logger.info(f"Processing reviews from {reviews_gz}...")
    logger.info(f"Output will be written to {output_csv}")
    logger.info(f"Filtering for {len(book_ids):,} unique book IDs")
    
    if not reviews_gz.exists():
        raise FileNotFoundError(f"Reviews file not found: {reviews_gz}")
    
    # Get file size for progress estimation
    file_size_mb = reviews_gz.stat().st_size / (1024 * 1024)
    logger.info(f"Reviews file size: {file_size_mb:.2f} MB")
    
    # Statistics
    total_reviews_processed = 0
    reviews_matched = 0
    reviews_english = 0
    reviews_written = 0
    json_errors = 0
    lang_detection_errors = 0
    empty_reviews = 0
    
    # Timing
    start_time = time.time()
    last_progress_time = start_time
    
    # Create output directory if it doesn't exist
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting review extraction...")
    
    # Open output CSV file
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['review_id', 'review_text', 'rating', 'book_id']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process reviews line by line
        with gzip.open(reviews_gz, 'rt', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                total_reviews_processed += 1
                
                # Parse JSON
                try:
                    review = json.loads(line)
                except json.JSONDecodeError as e:
                    json_errors += 1
                    if json_errors <= 10:  # Only log first 10 errors
                        logger.warning(f"JSON decode error at line {line_num}: {e}")
                    continue
                
                # Check if book_id matches
                book_id = review.get('book_id', '')
                if str(book_id) not in book_ids:
                    continue
                
                reviews_matched += 1
                
                # Get review text
                review_text = review.get('review_text', '')
                if not review_text:
                    empty_reviews += 1
                    continue
                
                # Check if English
                try:
                    if not is_english_review(review_text):
                        continue
                except Exception as e:
                    lang_detection_errors += 1
                    if lang_detection_errors <= 10:  # Only log first 10 errors
                        logger.warning(f"Language detection error at line {line_num}: {e}")
                    continue
                
                reviews_english += 1
                
                # Get rating (may be missing)
                rating = review.get('rating', '')
                if rating == '' or rating is None:
                    rating = ''
                else:
                    rating = str(rating)
                
                # Write to CSV
                writer.writerow({
                    'review_id': review.get('review_id', ''),
                    'review_text': review_text,
                    'rating': rating,
                    'book_id': book_id
                })
                
                reviews_written += 1
                
                # Progress logging with rate and time estimates
                if total_reviews_processed % PROGRESS_INTERVAL == 0:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    interval_elapsed = current_time - last_progress_time
                    
                    # Calculate rates
                    reviews_per_sec = total_reviews_processed / elapsed if elapsed > 0 else 0
                    matched_rate = (reviews_matched / total_reviews_processed * 100) if total_reviews_processed > 0 else 0
                    english_rate = (reviews_english / reviews_matched * 100) if reviews_matched > 0 else 0
                    write_rate = (reviews_written / reviews_english * 100) if reviews_english > 0 else 0
                    
                    # Estimate time remaining (rough estimate based on current rate)
                    # This is approximate since we don't know total lines
                    interval_rate = PROGRESS_INTERVAL / interval_elapsed if interval_elapsed > 0 else 0
                    
                    logger.info(
                        f"Progress: {total_reviews_processed:,} processed | "
                        f"Rate: {reviews_per_sec:.1f} reviews/sec | "
                        f"Matched: {reviews_matched:,} ({matched_rate:.2f}%) | "
                        f"English: {reviews_english:,} ({english_rate:.2f}% of matched) | "
                        f"Written: {reviews_written:,} ({write_rate:.2f}% of English) | "
                        f"Elapsed: {elapsed/60:.1f} min"
                    )
                    
                    last_progress_time = current_time
    
    # Final statistics
    total_elapsed = time.time() - start_time
    avg_rate = total_reviews_processed / total_elapsed if total_elapsed > 0 else 0
    
    logger.info("=" * 80)
    logger.info("Extraction complete!")
    logger.info("=" * 80)
    logger.info(f"Total reviews processed: {total_reviews_processed:,}")
    logger.info(f"Reviews matching book IDs: {reviews_matched:,} ({reviews_matched/total_reviews_processed*100:.2f}%)")
    logger.info(f"English reviews found: {reviews_english:,} ({reviews_english/reviews_matched*100:.2f}% of matched)")
    logger.info(f"Reviews written to CSV: {reviews_written:,}")
    logger.info(f"Empty reviews skipped: {empty_reviews:,}")
    logger.info(f"JSON decode errors: {json_errors}")
    logger.info(f"Language detection errors: {lang_detection_errors}")
    logger.info(f"Average processing rate: {avg_rate:.1f} reviews/second")
    logger.info(f"Total time elapsed: {total_elapsed/60:.2f} minutes ({total_elapsed:.2f} seconds)")
    logger.info(f"Output file: {output_csv}")
    
    # Calculate output file size
    if output_csv.exists():
        output_size_mb = output_csv.stat().st_size / (1024 * 1024)
        logger.info(f"Output file size: {output_size_mb:.2f} MB")


def main():
    """Main function."""
    # Set up logging
    log_dir = PROJECT_ROOT / "logs"
    logger = setup_logging(log_dir)
    
    try:
        logger.info("=" * 80)
        logger.info("Starting review extraction process")
        logger.info("=" * 80)
        logger.info(f"Books CSV: {BOOKS_CSV}")
        logger.info(f"Reviews file: {REVIEWS_GZ}")
        logger.info(f"Output CSV: {OUTPUT_CSV}")
        
        # Load book IDs
        book_ids = load_book_ids(BOOKS_CSV, logger)
        
        if not book_ids:
            logger.error("No book IDs found in CSV file")
            sys.exit(1)
        
        # Extract reviews
        extract_reviews(book_ids, REVIEWS_GZ, OUTPUT_CSV, logger)
        
        logger.info("=" * 80)
        logger.info("Script completed successfully!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

