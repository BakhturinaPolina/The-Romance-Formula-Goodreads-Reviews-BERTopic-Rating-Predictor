#!/usr/bin/env python3
"""
Extract English reviews for romance books from Goodreads reviews dataset.

This script:
1. Reads book IDs from romance_books_main_final.csv
2. Filters reviews from goodreads_reviews_romance.json.gz matching those book IDs
3. Detects and keeps only English reviews using language detection
4. Outputs a single CSV with columns: review_id, review_text, rating, book_id
"""

import argparse
import ast
import csv
import gzip
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Set, Optional

import pandas as pd
from langdetect import detect, LangDetectException

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_BOOKS_CSV = Path("/home/polina/Documents/goodreads_romance_research_cursor/anna_archive_romance_pipeline/data/processed/romance_books_main_final.csv")
DEFAULT_REVIEWS_GZ = PROJECT_ROOT / "data" / "raw" / "goodreads_reviews_romance.json.gz"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "romance_reviews_english.csv"

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


def load_book_ids_from_work_ids(
    subdataset_csv: Path,
    main_dataset_csv: Path,
    logger: logging.Logger
) -> Set[str]:
    """
    Load book IDs by mapping work_ids from subdataset to book_id_list_en from main dataset.
    
    Args:
        subdataset_csv: Path to subdataset CSV with work_id column
        main_dataset_csv: Path to main dataset CSV with work_id and book_id_list_en columns
        logger: Logger instance for logging
        
    Returns:
        Set of book IDs as strings
    """
    logger.info(f"Loading work_ids from subdataset: {subdataset_csv}...")
    start_time = time.time()
    
    # Load subdataset
    sub_df = pd.read_csv(subdataset_csv)
    logger.info(f"Subdataset loaded: {len(sub_df)} rows, {len(sub_df.columns)} columns")
    
    if 'work_id' not in sub_df.columns:
        raise ValueError(f"Column 'work_id' not found in subdataset. Available columns: {sub_df.columns.tolist()}")
    
    work_ids = set(sub_df['work_id'].dropna().astype(str))
    logger.info(f"Found {len(work_ids):,} unique work_ids in subdataset")
    
    # Load main dataset
    logger.info(f"Loading main dataset for mapping: {main_dataset_csv}...")
    main_df = pd.read_csv(main_dataset_csv)
    logger.info(f"Main dataset loaded: {len(main_df)} rows")
    
    if 'work_id' not in main_df.columns or 'book_id_list_en' not in main_df.columns:
        raise ValueError(f"Main dataset must have 'work_id' and 'book_id_list_en' columns. Available: {main_df.columns.tolist()}")
    
    # Create mapping from work_id to book_id_list_en
    work_id_to_book_ids = {}
    for idx, row in main_df.iterrows():
        work_id = str(row['work_id'])
        if work_id in work_ids:
            book_id_list_str = row['book_id_list_en']
            if pd.notna(book_id_list_str) and book_id_list_str and str(book_id_list_str).strip():
                try:
                    book_id_list = ast.literal_eval(str(book_id_list_str))
                    work_id_to_book_ids[work_id] = book_id_list
                except (ValueError, SyntaxError):
                    continue
    
    logger.info(f"Mapped {len(work_id_to_book_ids):,} work_ids to book_id_list_en")
    
    # Collect all book IDs
    book_ids = set()
    for work_id, book_id_list in work_id_to_book_ids.items():
        for book_id in book_id_list:
            book_ids.add(str(book_id))
    
    elapsed = time.time() - start_time
    logger.info(f"Loaded {len(book_ids):,} unique book IDs from {len(work_id_to_book_ids):,} works in {elapsed:.2f} seconds")
    
    return book_ids


def load_book_ids(csv_path: Path, logger: logging.Logger) -> Set[str]:
    """
    Load book IDs from the CSV file (direct mode with book_id_list_en column).
    
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


def load_existing_review_ids(output_csv: Path, logger: logging.Logger) -> Set[str]:
    """
    Load existing review IDs from output CSV file for resume functionality.
    
    Args:
        output_csv: Path to output CSV file
        logger: Logger instance for logging
        
    Returns:
        Set of existing review IDs
    """
    if not output_csv.exists():
        return set()
    
    logger.info(f"Loading existing review IDs from {output_csv} for resume...")
    start_time = time.time()
    
    existing_review_ids = set()
    try:
        with open(output_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                review_id = row.get('review_id', '').strip()
                if review_id:
                    existing_review_ids.add(review_id)
        
        elapsed = time.time() - start_time
        logger.info(f"Loaded {len(existing_review_ids):,} existing review IDs in {elapsed:.2f} seconds")
        logger.info(f"Resume mode: Will skip {len(existing_review_ids):,} already extracted reviews")
    except Exception as e:
        logger.warning(f"Could not load existing review IDs: {e}. Starting fresh.")
        return set()
    
    return existing_review_ids


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
    Supports resume functionality by skipping already extracted reviews.
    
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
    
    # Load existing review IDs for resume functionality
    existing_review_ids = load_existing_review_ids(output_csv, logger)
    resume_mode = len(existing_review_ids) > 0
    
    # Statistics
    total_reviews_processed = 0
    reviews_matched = 0
    reviews_english = 0
    reviews_written = 0
    reviews_skipped = 0  # Already in output file
    json_errors = 0
    lang_detection_errors = 0
    empty_reviews = 0
    
    # Timing
    start_time = time.time()
    last_progress_time = start_time
    
    # Create output directory if it doesn't exist
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    if resume_mode:
        logger.info("Resume mode: Appending to existing file")
    else:
        logger.info("Starting fresh extraction...")
    
    # Determine file mode: append if resuming, write if starting fresh
    file_mode = 'a' if resume_mode else 'w'
    
    # Open output CSV file
    with open(output_csv, file_mode, newline='', encoding='utf-8') as csvfile:
        fieldnames = ['review_id', 'review_text', 'rating', 'book_id']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Only write header if starting fresh (not in resume mode)
        if not resume_mode:
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
                
                # Check if review already exists (resume functionality)
                review_id = review.get('review_id', '')
                if review_id in existing_review_ids:
                    reviews_skipped += 1
                    continue
                
                # Get rating (may be missing)
                rating = review.get('rating', '')
                if rating == '' or rating is None:
                    rating = ''
                else:
                    rating = str(rating)
                
                # Write to CSV
                writer.writerow({
                    'review_id': review_id,
                    'review_text': review_text,
                    'rating': rating,
                    'book_id': book_id
                })
                
                reviews_written += 1
                # Add to existing set to avoid duplicates in same run
                existing_review_ids.add(review_id)
                
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
                    
                    skip_info = f" | Skipped: {reviews_skipped:,}" if reviews_skipped > 0 else ""
                    logger.info(
                        f"Progress: {total_reviews_processed:,} processed | "
                        f"Rate: {reviews_per_sec:.1f} reviews/sec | "
                        f"Matched: {reviews_matched:,} ({matched_rate:.2f}%) | "
                        f"English: {reviews_english:,} ({english_rate:.2f}% of matched) | "
                        f"Written: {reviews_written:,} ({write_rate:.2f}% of English){skip_info} | "
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
    if reviews_skipped > 0:
        logger.info(f"Reviews skipped (already existed): {reviews_skipped:,}")
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
    parser = argparse.ArgumentParser(
        description='Extract English reviews for romance books from Goodreads reviews dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract reviews for full dataset (default)
  python3 extract_reviews.py
  
  # Extract reviews for subdataset using work_id mapping
  python3 extract_reviews.py \\
    --subdataset data/processed/romance_subdataset_6000.csv \\
    --main-dataset /path/to/romance_books_main_final.csv \\
    --output data/processed/romance_reviews_english_subdataset_6000.csv
  
  # Custom input/output paths
  python3 extract_reviews.py \\
    --books-csv /path/to/books.csv \\
    --output /path/to/output.csv
        """
    )
    
    parser.add_argument(
        '--books-csv',
        type=Path,
        default=None,
        help=f'Path to books CSV with book_id_list_en column (default: {DEFAULT_BOOKS_CSV})'
    )
    parser.add_argument(
        '--subdataset',
        type=Path,
        default=None,
        help='Path to subdataset CSV with work_id column (enables work_id mapping mode)'
    )
    parser.add_argument(
        '--main-dataset',
        type=Path,
        default=DEFAULT_BOOKS_CSV,
        help=f'Path to main dataset CSV for work_id mapping (default: {DEFAULT_BOOKS_CSV})'
    )
    parser.add_argument(
        '--reviews-file',
        type=Path,
        default=DEFAULT_REVIEWS_GZ,
        help=f'Path to gzipped reviews JSON file (default: {DEFAULT_REVIEWS_GZ})'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Path to output CSV file (default: based on input file name)'
    )
    
    args = parser.parse_args()
    
    # Determine mode and set paths
    use_work_id_mapping = args.subdataset is not None
    
    if use_work_id_mapping:
        books_csv = args.subdataset
        if args.output is None:
            # Generate output filename from subdataset name
            subdataset_name = args.subdataset.stem
            args.output = PROJECT_ROOT / "data" / "processed" / f"romance_reviews_english_{subdataset_name}.csv"
    else:
        books_csv = args.books_csv or DEFAULT_BOOKS_CSV
        if args.output is None:
            args.output = DEFAULT_OUTPUT_CSV
    
    reviews_gz = args.reviews_file
    output_csv = args.output
    
    # Set up logging
    log_dir = PROJECT_ROOT / "logs"
    logger = setup_logging(log_dir)
    
    try:
        logger.info("=" * 80)
        logger.info("Starting review extraction process")
        logger.info("=" * 80)
        logger.info(f"Mode: {'work_id mapping (subdataset)' if use_work_id_mapping else 'direct (book_id_list_en)'}")
        logger.info(f"Books CSV: {books_csv}")
        if use_work_id_mapping:
            logger.info(f"Main dataset: {args.main_dataset}")
        logger.info(f"Reviews file: {reviews_gz}")
        logger.info(f"Output CSV: {output_csv}")
        
        # Load book IDs
        if use_work_id_mapping:
            book_ids = load_book_ids_from_work_ids(books_csv, args.main_dataset, logger)
        else:
            book_ids = load_book_ids(books_csv, logger)
        
        if not book_ids:
            logger.error("No book IDs found in CSV file")
            sys.exit(1)
        
        # Extract reviews
        extract_reviews(book_ids, reviews_gz, output_csv, logger)
        
        logger.info("=" * 80)
        logger.info("Script completed successfully!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

