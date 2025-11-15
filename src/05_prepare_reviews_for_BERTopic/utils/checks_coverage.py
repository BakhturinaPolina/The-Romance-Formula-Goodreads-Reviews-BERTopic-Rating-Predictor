"""
Coverage checking module for review-based topic modeling pipeline.

This module computes review coverage statistics for all books in the dataset,
including review counts per book and distribution by quality tier (pop_tier).
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional

# Import data loading functions
try:
    from .data_loading import load_books, load_reviews, load_joined_reviews
except ImportError:
    # If running as script, add parent directory to path and import
    import sys
    from pathlib import Path
    # Add the utils directory to path
    utils_dir = Path(__file__).parent
    if str(utils_dir) not in sys.path:
        sys.path.insert(0, str(utils_dir))
    from data_loading import load_books, load_reviews, load_joined_reviews

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_INTERIM = PROJECT_ROOT / "data" / "intermediate"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def compute_review_counts(
    books_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    main_dataset_file: Optional[Path] = None
) -> pd.DataFrame:
    """
    Compute review counts per book by joining reviews to books.
    
    Args:
        books_df: DataFrame with books (must have 'work_id' column)
        reviews_df: DataFrame with reviews (must have 'book_id' column)
        main_dataset_file: Path to main dataset for book_id -> work_id mapping
    
    Returns:
        DataFrame with books and review counts:
        - All original book columns
        - n_reviews_english: Count of English reviews per book
        - has_reviews: Boolean indicating if book has any reviews
    """
    logger.info("Computing review counts per book...")
    
    # Load the joined dataset (this handles the mapping internally)
    joined_df = load_joined_reviews(
        books_file=None,  # Will use default
        reviews_file=None,  # Will use default
        main_dataset_file=main_dataset_file,
        how="left"  # Left join to keep all books, even without reviews
    )
    
    # Group by work_id and count reviews
    review_counts = (
        joined_df
        .groupby('work_id')
        .agg({
            'review_id': 'count',  # Count of reviews
            'work_id': 'first'  # Keep work_id
        })
        .rename(columns={'review_id': 'n_reviews_english'})
        .reset_index(drop=True)
    )
    
    # Merge back to books dataframe
    coverage_df = books_df.merge(
        review_counts,
        on='work_id',
        how='left'
    )
    
    # Fill NaN with 0 for books without reviews
    coverage_df['n_reviews_english'] = coverage_df['n_reviews_english'].fillna(0).astype(int)
    coverage_df['has_reviews'] = coverage_df['n_reviews_english'] > 0
    
    logger.info(f"Computed review counts for {len(coverage_df):,} books")
    logger.info(f"Books with reviews: {coverage_df['has_reviews'].sum():,}")
    logger.info(f"Books without reviews: {(~coverage_df['has_reviews']).sum():,}")
    
    return coverage_df


def generate_coverage_table(
    books_file: Optional[Path] = None,
    reviews_file: Optional[Path] = None,
    main_dataset_file: Optional[Path] = None,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Generate complete coverage table with review counts per book.
    
    Args:
        books_file: Path to books CSV. If None, uses default location.
        reviews_file: Path to reviews CSV. If None, uses default location.
        main_dataset_file: Path to main dataset for mapping. If None, uses default.
        output_path: Path to save coverage table. If None, uses default location.
    
    Returns:
        DataFrame with coverage information
    """
    logger.info("=" * 80)
    logger.info("Generating review coverage table")
    logger.info("=" * 80)
    
    # Load books
    books_df = load_books(file_path=books_file)
    
    # Load reviews
    reviews_df = load_reviews(file_path=reviews_file)
    
    # Compute review counts
    coverage_df = compute_review_counts(
        books_df=books_df,
        reviews_df=reviews_df,
        main_dataset_file=main_dataset_file
    )
    
    # Save to file
    if output_path is None:
        DATA_INTERIM.mkdir(parents=True, exist_ok=True)
        output_path = DATA_INTERIM / "review_coverage_by_book.csv"
    
    logger.info(f"Saving coverage table to: {output_path}")
    coverage_df.to_csv(output_path, index=False)
    logger.info(f"Coverage table saved: {len(coverage_df):,} rows")
    
    return coverage_df


def summarize_coverage(
    coverage_df: pd.DataFrame,
    print_summary: bool = True
) -> dict:
    """
    Generate summary statistics about review coverage.
    
    Args:
        coverage_df: DataFrame with coverage information (from generate_coverage_table)
        print_summary: If True, print summary to console
    
    Returns:
        Dictionary with summary statistics
    """
    logger.info("Computing coverage summary statistics...")
    
    summary = {}
    
    # Overall statistics
    summary['total_books'] = len(coverage_df)
    summary['books_with_reviews'] = coverage_df['has_reviews'].sum()
    summary['books_without_reviews'] = (~coverage_df['has_reviews']).sum()
    summary['coverage_rate'] = summary['books_with_reviews'] / summary['total_books']
    
    # Review count statistics
    summary['total_reviews'] = coverage_df['n_reviews_english'].sum()
    summary['mean_reviews_per_book'] = coverage_df['n_reviews_english'].mean()
    summary['median_reviews_per_book'] = coverage_df['n_reviews_english'].median()
    summary['min_reviews'] = coverage_df['n_reviews_english'].min()
    summary['max_reviews'] = coverage_df['n_reviews_english'].max()
    
    # Statistics by pop_tier
    if 'pop_tier' in coverage_df.columns:
        tier_stats = {}
        for tier in coverage_df['pop_tier'].unique():
            tier_df = coverage_df[coverage_df['pop_tier'] == tier]
            tier_stats[tier] = {
                'total_books': len(tier_df),
                'books_with_reviews': tier_df['has_reviews'].sum(),
                'books_without_reviews': (~tier_df['has_reviews']).sum(),
                'coverage_rate': tier_df['has_reviews'].sum() / len(tier_df),
                'total_reviews': tier_df['n_reviews_english'].sum(),
                'mean_reviews': tier_df['n_reviews_english'].mean(),
                'median_reviews': tier_df['n_reviews_english'].median(),
                'min_reviews': tier_df['n_reviews_english'].min(),
                'max_reviews': tier_df['n_reviews_english'].max()
            }
        summary['by_pop_tier'] = tier_stats
    
    # Distribution percentiles
    summary['percentiles'] = {
        'p10': coverage_df['n_reviews_english'].quantile(0.10),
        'p25': coverage_df['n_reviews_english'].quantile(0.25),
        'p50': coverage_df['n_reviews_english'].quantile(0.50),
        'p75': coverage_df['n_reviews_english'].quantile(0.75),
        'p90': coverage_df['n_reviews_english'].quantile(0.90),
        'p95': coverage_df['n_reviews_english'].quantile(0.95),
        'p99': coverage_df['n_reviews_english'].quantile(0.99)
    }
    
    # Print summary if requested
    if print_summary:
        print("\n" + "=" * 80)
        print("REVIEW COVERAGE SUMMARY")
        print("=" * 80)
        print(f"\nOverall Statistics:")
        print(f"  Total books: {summary['total_books']:,}")
        print(f"  Books with reviews: {summary['books_with_reviews']:,} ({summary['coverage_rate']*100:.2f}%)")
        print(f"  Books without reviews: {summary['books_without_reviews']:,}")
        print(f"  Total reviews: {summary['total_reviews']:,}")
        print(f"  Mean reviews per book: {summary['mean_reviews_per_book']:.1f}")
        print(f"  Median reviews per book: {summary['median_reviews_per_book']:.1f}")
        print(f"  Min reviews: {summary['min_reviews']}")
        print(f"  Max reviews: {summary['max_reviews']:,}")
        
        print(f"\nReview Count Distribution (Percentiles):")
        for p_name, p_value in summary['percentiles'].items():
            print(f"  {p_name}: {p_value:.1f}")
        
        if 'by_pop_tier' in summary:
            print(f"\nStatistics by Pop Tier:")
            for tier, stats in summary['by_pop_tier'].items():
                print(f"\n  {tier.upper()}:")
                print(f"    Total books: {stats['total_books']:,}")
                print(f"    Books with reviews: {stats['books_with_reviews']:,} ({stats['coverage_rate']*100:.2f}%)")
                print(f"    Total reviews: {stats['total_reviews']:,}")
                print(f"    Mean reviews per book: {stats['mean_reviews']:.1f}")
                print(f"    Median reviews per book: {stats['median_reviews']:.1f}")
                print(f"    Range: {stats['min_reviews']} - {stats['max_reviews']:,}")
        
        print("\n" + "=" * 80)
    
    return summary


def save_coverage_table(
    coverage_df: pd.DataFrame,
    output_path: Optional[Path] = None
) -> Path:
    """
    Save coverage table to CSV file.
    
    Args:
        coverage_df: DataFrame with coverage information
        output_path: Path to save file. If None, uses default location.
    
    Returns:
        Path to saved file
    """
    if output_path is None:
        DATA_INTERIM.mkdir(parents=True, exist_ok=True)
        output_path = DATA_INTERIM / "review_coverage_by_book.csv"
    
    logger.info(f"Saving coverage table to: {output_path}")
    coverage_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(coverage_df):,} rows to {output_path}")
    
    return output_path


if __name__ == "__main__":
    # Example usage
    print("Generating review coverage table...")
    
    # Generate coverage table
    coverage_df = generate_coverage_table()
    
    # Generate and print summary
    summary = summarize_coverage(coverage_df, print_summary=True)
    
    print(f"\nCoverage table saved with {len(coverage_df):,} books")
    print(f"Coverage rate: {summary['coverage_rate']*100:.2f}%")

