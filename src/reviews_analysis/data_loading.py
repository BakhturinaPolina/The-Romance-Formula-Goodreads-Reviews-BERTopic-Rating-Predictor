"""
Data loading module for review-based topic modeling pipeline.

This module provides functions to load books and reviews datasets,
and join them together for analysis.
"""

import pandas as pd
import logging
import ast
from pathlib import Path
from typing import Optional, Tuple, Dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def load_books(
    file_path: Optional[Path] = None,
    expected_columns: Optional[list] = None
) -> pd.DataFrame:
    """
    Load the books dataset (6,000 romance novels with quality flags).
    
    Args:
        file_path: Path to the books CSV file. If None, uses default location.
        expected_columns: List of expected column names for validation.
                         If None, uses default expected columns.
    
    Returns:
        DataFrame with books data including:
        - work_id: Unique book identifier (join key)
        - pop_tier: Quality classification (trash/middle/top)
        - title, author_name, publication_year, etc.
    
    Raises:
        FileNotFoundError: If the books file doesn't exist.
        ValueError: If required columns are missing.
    """
    if file_path is None:
        file_path = DATA_PROCESSED / "romance_subdataset_6000.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Books file not found: {file_path}")
    
    logger.info(f"Loading books dataset from: {file_path}")
    books_df = pd.read_csv(file_path)
    
    # Default expected columns
    if expected_columns is None:
        expected_columns = [
            "work_id", "title", "author_id", "author_name", "publication_year",
            "num_pages_median", "genres_str", "series_id", "series_title",
            "ratings_count_sum", "text_reviews_count_sum",
            "average_rating_weighted_mean", "pop_tier"
        ]
    
    # Validate required columns
    missing_cols = set(expected_columns) - set(books_df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns in books dataset: {missing_cols}"
        )
    
    # Validate join key exists
    if "work_id" not in books_df.columns:
        raise ValueError("books dataset must contain 'work_id' column for joining")
    
    # Check for duplicates in work_id
    n_duplicates = books_df["work_id"].duplicated().sum()
    if n_duplicates > 0:
        logger.warning(
            f"Found {n_duplicates} duplicate work_id values in books dataset"
        )
    
    # Log basic statistics
    logger.info(f"Loaded {len(books_df):,} books")
    logger.info(f"Pop tier distribution:\n{books_df['pop_tier'].value_counts()}")
    
    return books_df


def load_reviews(
    file_path: Optional[Path] = None,
    expected_columns: Optional[list] = None
) -> pd.DataFrame:
    """
    Load the English reviews dataset.
    
    Args:
        file_path: Path to the reviews CSV file. If None, uses default location.
        expected_columns: List of expected column names for validation.
                         If None, uses default expected columns.
    
    Returns:
        DataFrame with reviews data including:
        - review_id: Unique review identifier
        - book_id: Foreign key to books (maps to work_id)
        - review_text: The actual review content (English)
        - rating: Star rating (if available)
    
    Raises:
        FileNotFoundError: If the reviews file doesn't exist.
        ValueError: If required columns are missing.
    """
    if file_path is None:
        # Try subdataset-specific file first, then fall back to general file
        subdataset_file = DATA_PROCESSED / "romance_reviews_english_subdataset_6000.csv"
        general_file = DATA_PROCESSED / "romance_reviews_english.csv"
        
        if subdataset_file.exists():
            file_path = subdataset_file
        elif general_file.exists():
            file_path = general_file
        else:
            raise FileNotFoundError(
                f"Reviews file not found. Tried: {subdataset_file}, {general_file}"
            )
    
    if not file_path.exists():
        raise FileNotFoundError(f"Reviews file not found: {file_path}")
    
    logger.info(f"Loading reviews dataset from: {file_path}")
    reviews_df = pd.read_csv(file_path)
    
    # Default expected columns
    if expected_columns is None:
        expected_columns = ["review_id", "review_text", "rating", "book_id"]
    
    # Validate required columns
    missing_cols = set(expected_columns) - set(reviews_df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns in reviews dataset: {missing_cols}"
        )
    
    # Validate join key exists
    if "book_id" not in reviews_df.columns:
        raise ValueError("reviews dataset must contain 'book_id' column for joining")
    
    # Check for missing review text
    n_missing_text = reviews_df["review_text"].isna().sum()
    if n_missing_text > 0:
        logger.warning(
            f"Found {n_missing_text:,} reviews with missing review_text"
        )
    
    # Log basic statistics
    logger.info(f"Loaded {len(reviews_df):,} reviews")
    logger.info(f"Unique books in reviews: {reviews_df['book_id'].nunique():,}")
    
    return reviews_df


def load_book_id_to_work_id_mapping(
    main_dataset_file: Optional[Path] = None,
    subdataset_work_ids: Optional[pd.Series] = None
) -> Dict[int, int]:
    """
    Load mapping from book_id to work_id using the main dataset.
    
    In Goodreads:
    - work_id: Represents a work (abstract book concept)
    - book_id: Represents a specific edition (paperback, hardcover, etc.)
    - book_id_list_en: List of all book_ids (editions) for a work_id
    
    Args:
        main_dataset_file: Path to main dataset CSV with book_id_list_en.
                          If None, tries default location.
        subdataset_work_ids: Series of work_ids to filter mapping to.
                           If None, creates mapping for all works in main dataset.
    
    Returns:
        Dictionary mapping book_id -> work_id
    
    Raises:
        FileNotFoundError: If main dataset file doesn't exist.
    """
    if main_dataset_file is None:
        # Try default locations
        possible_locations = [
            PROJECT_ROOT / "data" / "processed" / "romance_books_main_final.csv",
            Path("/home/polina/Documents/goodreads_romance_research_cursor/anna_archive_romance_pipeline/data/processed/romance_books_main_final.csv")
        ]
        
        for loc in possible_locations:
            if loc.exists():
                main_dataset_file = loc
                break
        
        if main_dataset_file is None or not main_dataset_file.exists():
            raise FileNotFoundError(
                f"Main dataset not found. Tried: {possible_locations}"
            )
    
    logger.info(f"Loading book_id -> work_id mapping from: {main_dataset_file}")
    
    # Load only necessary columns
    main_df = pd.read_csv(
        main_dataset_file,
        usecols=['work_id', 'book_id_list_en']
    )
    
    # Filter to subdataset work_ids if provided
    if subdataset_work_ids is not None:
        work_ids_set = set(subdataset_work_ids.unique())
        main_df = main_df[main_df['work_id'].isin(work_ids_set)]
        logger.info(f"Filtered to {len(main_df):,} works from subdataset")
    
    # Create mapping: book_id -> work_id
    book_to_work = {}
    parse_errors = 0
    
    for idx, row in main_df.iterrows():
        work_id = int(row['work_id'])
        book_id_list_str = row['book_id_list_en']
        
        if pd.isna(book_id_list_str) or not book_id_list_str:
            continue
        
        try:
            book_id_list = ast.literal_eval(book_id_list_str)
            for bid in book_id_list:
                try:
                    bid_int = int(bid)
                    book_to_work[bid_int] = work_id
                except (ValueError, TypeError):
                    continue
        except (ValueError, SyntaxError) as e:
            parse_errors += 1
            if parse_errors <= 5:
                logger.warning(f"Could not parse book_id_list_en for work_id {work_id}: {e}")
            continue
    
    logger.info(
        f"Created mapping: {len(book_to_work):,} book_ids -> {main_df['work_id'].nunique():,} work_ids"
    )
    if parse_errors > 0:
        logger.warning(f"Encountered {parse_errors} parse errors")
    
    return book_to_work


def load_joined_reviews(
    books_file: Optional[Path] = None,
    reviews_file: Optional[Path] = None,
    main_dataset_file: Optional[Path] = None,
    how: str = "inner"
) -> pd.DataFrame:
    """
    Load and join books and reviews datasets using work_id -> book_id mapping.
    
    Args:
        books_file: Path to books CSV. If None, uses default location.
        reviews_file: Path to reviews CSV. If None, uses default location.
        main_dataset_file: Path to main dataset with book_id_list_en for mapping.
                          If None, tries default locations.
        how: Type of join ('inner', 'left', 'right', 'outer').
             Default is 'inner' to only keep books with reviews.
    
    Returns:
        DataFrame with joined books and reviews data.
        Join is performed by:
        1. Mapping book_id (reviews) -> work_id using main dataset
        2. Joining on work_id
    
    Raises:
        FileNotFoundError: If required files don't exist.
        ValueError: If join fails due to missing columns or ID mismatches.
    """
    # Load both datasets
    books_df = load_books(file_path=books_file)
    reviews_df = load_reviews(file_path=reviews_file)
    
    logger.info("Joining books and reviews datasets...")
    logger.info(f"  Books: {len(books_df):,} records")
    logger.info(f"  Reviews: {len(reviews_df):,} records")
    
    # Load mapping from book_id to work_id
    book_to_work = load_book_id_to_work_id_mapping(
        main_dataset_file=main_dataset_file,
        subdataset_work_ids=books_df['work_id']
    )
    
    # Map book_id to work_id in reviews
    logger.info("Mapping book_id to work_id in reviews...")
    reviews_df = reviews_df.copy()
    reviews_df['work_id'] = reviews_df['book_id'].map(book_to_work)
    
    # Count how many reviews got mapped
    n_mapped = reviews_df['work_id'].notna().sum()
    n_unmapped = reviews_df['work_id'].isna().sum()
    logger.info(
        f"  Mapped: {n_mapped:,} reviews ({n_mapped/len(reviews_df)*100:.1f}%)"
    )
    if n_unmapped > 0:
        logger.warning(
            f"  Unmapped: {n_unmapped:,} reviews ({n_unmapped/len(reviews_df)*100:.1f}%)"
        )
    
    # Perform the join on work_id
    joined_df = reviews_df.merge(
        books_df,
        on="work_id",
        how=how,
        suffixes=("_review", "_book")
    )
    
    # Log join results
    logger.info(f"Joined dataset: {len(joined_df):,} records")
    
    if how == "inner":
        n_books_with_reviews = joined_df["work_id"].nunique()
        logger.info(f"Books with reviews: {n_books_with_reviews:,}")
    elif how == "left":
        n_books_with_reviews = joined_df["work_id"].nunique()
        n_books_total = books_df["work_id"].nunique()
        logger.info(
            f"Books with reviews: {n_books_with_reviews:,} / {n_books_total:,} "
            f"({n_books_with_reviews/n_books_total*100:.1f}%)"
        )
    
    return joined_df


def validate_join_keys(
    books_df: pd.DataFrame,
    reviews_df: pd.DataFrame
) -> dict:
    """
    Validate the join keys between books and reviews datasets.
    
    Args:
        books_df: Books DataFrame (must have 'work_id' column)
        reviews_df: Reviews DataFrame (must have 'book_id' column)
    
    Returns:
        Dictionary with validation results:
        - books_total: Total number of books
        - reviews_total: Total number of reviews
        - books_with_reviews: Number of books that have at least one review
        - reviews_matched: Number of reviews that match a book
        - reviews_unmatched: Number of reviews that don't match any book
        - books_without_reviews: Number of books without any reviews
    """
    books_ids = set(books_df["work_id"].unique())
    reviews_ids = set(reviews_df["book_id"].unique())
    
    matched_ids = books_ids & reviews_ids
    books_without_reviews = books_ids - reviews_ids
    reviews_unmatched = reviews_ids - books_ids
    
    reviews_matched = reviews_df[
        reviews_df["book_id"].isin(matched_ids)
    ].shape[0]
    
    return {
        "books_total": len(books_ids),
        "reviews_total": len(reviews_df),
        "books_with_reviews": len(matched_ids),
        "reviews_matched": reviews_matched,
        "reviews_unmatched": len(reviews_unmatched),
        "books_without_reviews": len(books_without_reviews),
        "match_rate_books": len(matched_ids) / len(books_ids) if books_ids else 0,
        "match_rate_reviews": reviews_matched / len(reviews_df) if len(reviews_df) > 0 else 0
    }


if __name__ == "__main__":
    # Example usage and testing
    print("Testing data loading functions...")
    
    # Load books
    books = load_books()
    print(f"\nBooks dataset shape: {books.shape}")
    print(f"Books columns: {list(books.columns)}")
    
    # Load reviews
    reviews = load_reviews()
    print(f"\nReviews dataset shape: {reviews.shape}")
    print(f"Reviews columns: {list(reviews.columns)}")
    
    # Validate join keys
    validation = validate_join_keys(books, reviews)
    print("\nJoin key validation:")
    for key, value in validation.items():
        print(f"  {key}: {value}")
    
    # Load joined dataset
    joined = load_joined_reviews()
    print(f"\nJoined dataset shape: {joined.shape}")
    print(f"Joined columns: {list(joined.columns)}")
    print(f"\nSample of joined data:")
    print(joined[['work_id', 'title', 'pop_tier', 'review_text']].head(3))

