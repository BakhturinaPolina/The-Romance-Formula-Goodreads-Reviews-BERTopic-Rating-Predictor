"""
Sample test datasets from the main review sentences parquet file.

Creates smaller datasets for quick testing of BERTopic+OCTIS pipeline.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project paths (go up 4 levels: file -> scripts -> 06_topic_modeling -> src -> project_root)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_INTERIM = PROJECT_ROOT / "data" / "intermediate"


def sample_test_dataset(
    input_parquet: Path,
    output_parquet: Path,
    n_samples: int = 10000,
    stratify_by: Optional[str] = 'pop_tier',
    seed: int = 42,
    preserve_work_ids: bool = True
) -> pd.DataFrame:
    """
    Sample sentences from the main dataset for testing.
    
    Args:
        input_parquet: Path to main review_sentences_for_bertopic.parquet
        output_parquet: Path to save sampled dataset
        n_samples: Number of sentences to sample
        stratify_by: Column to stratify by (e.g., 'pop_tier'). If None, random sampling.
        seed: Random seed for reproducibility
        preserve_work_ids: If True, samples entire reviews (keeps all sentences from sampled reviews)
    
    Returns:
        Sampled DataFrame
    """
    logger.info("=" * 80)
    logger.info("Sampling Test Dataset")
    logger.info("=" * 80)
    
    # Load main dataset
    logger.info(f"Loading main dataset from: {input_parquet}")
    if not input_parquet.exists():
        raise FileNotFoundError(f"Input file not found: {input_parquet}")
    
    df = pd.read_parquet(input_parquet)
    logger.info(f"  Loaded {len(df):,} sentences")
    logger.info(f"  Columns: {list(df.columns)}")
    
    # Show distribution
    if 'pop_tier' in df.columns:
        logger.info(f"\nPop tier distribution:")
        tier_counts = df['pop_tier'].value_counts()
        for tier, count in tier_counts.items():
            pct = count / len(df) * 100
            logger.info(f"  {tier}: {count:,} ({pct:.1f}%)")
    
    # Sampling strategy
    if preserve_work_ids:
        # Sample by review_id to keep all sentences from sampled reviews
        logger.info(f"\nSampling strategy: Preserve work_ids (sample entire reviews)")
        unique_reviews = df[['review_id', 'work_id']].drop_duplicates()
        
        if stratify_by and stratify_by in df.columns:
            # Get pop_tier for each review
            review_tiers = df[['review_id', stratify_by]].drop_duplicates()
            unique_reviews = unique_reviews.merge(review_tiers, on='review_id', how='left')
            
            # Stratified sampling by pop_tier
            logger.info(f"  Stratifying by: {stratify_by}")
            n_per_tier = n_samples // unique_reviews[stratify_by].nunique()
            sampled_reviews = unique_reviews.groupby(stratify_by).apply(
                lambda x: x.sample(min(len(x), n_per_tier), random_state=seed)
            ).reset_index(drop=True)
        else:
            # Random sampling
            sampled_reviews = unique_reviews.sample(n=min(len(unique_reviews), n_samples), random_state=seed)
        
        # Get all sentences from sampled reviews
        sampled_df = df[df['review_id'].isin(sampled_reviews['review_id'])].copy()
        logger.info(f"  Sampled {len(sampled_reviews):,} reviews")
        logger.info(f"  Result: {len(sampled_df):,} sentences")
        
    else:
        # Direct sentence sampling
        logger.info(f"\nSampling strategy: Direct sentence sampling")
        if stratify_by and stratify_by in df.columns:
            # Stratified sampling
            logger.info(f"  Stratifying by: {stratify_by}")
            n_per_tier = n_samples // df[stratify_by].nunique()
            sampled_df = df.groupby(stratify_by).apply(
                lambda x: x.sample(min(len(x), n_per_tier), random_state=seed)
            ).reset_index(drop=True)
        else:
            # Random sampling
            sampled_df = df.sample(n=min(len(df), n_samples), random_state=seed)
    
    # Reset sentence_id to be sequential
    sampled_df = sampled_df.reset_index(drop=True)
    sampled_df['sentence_id'] = range(len(sampled_df))
    
    # Show final distribution
    logger.info(f"\nFinal sample distribution:")
    logger.info(f"  Total sentences: {len(sampled_df):,}")
    if 'pop_tier' in sampled_df.columns:
        tier_counts = sampled_df['pop_tier'].value_counts()
        for tier, count in tier_counts.items():
            pct = count / len(sampled_df) * 100
            logger.info(f"  {tier}: {count:,} ({pct:.1f}%)")
    logger.info(f"  Unique reviews: {sampled_df['review_id'].nunique():,}")
    logger.info(f"  Unique works: {sampled_df['work_id'].nunique():,}")
    
    # Save sampled dataset
    logger.info(f"\nSaving sampled dataset to: {output_parquet}")
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    sampled_df.to_parquet(output_parquet, index=False)
    
    file_size_mb = output_parquet.stat().st_size / (1024 * 1024)
    logger.info(f"  ✓ Saved {len(sampled_df):,} sentences ({file_size_mb:.2f} MB)")
    
    return sampled_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sample test dataset from main review sentences")
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_PROCESSED / "review_sentences_for_bertopic.parquet",
        help="Input parquet file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_INTERIM / "review_sentences_test_10k.parquet",
        help="Output parquet file"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10000,
        help="Number of sentences to sample"
    )
    parser.add_argument(
        "--stratify",
        type=str,
        default="pop_tier",
        help="Column to stratify by (or 'none' for random sampling)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--preserve-reviews",
        action="store_true",
        help="Preserve entire reviews (sample by review_id)"
    )
    
    args = parser.parse_args()
    
    stratify_by = args.stratify if args.stratify.lower() != 'none' else None
    
    sample_test_dataset(
        input_parquet=args.input,
        output_parquet=args.output,
        n_samples=args.n_samples,
        stratify_by=stratify_by,
        seed=args.seed,
        preserve_work_ids=args.preserve_reviews
    )

