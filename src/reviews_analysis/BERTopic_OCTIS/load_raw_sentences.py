"""
Load raw (unpreprocessed) sentences from reviews for BERTopic embeddings.

This module loads review text and splits it into sentences WITHOUT cleaning,
as sentence transformer models need unpreprocessed text for optimal embeddings.
"""

import pandas as pd
import spacy
import logging
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"


def load_raw_sentences_from_reviews(
    reviews_file: Optional[Path] = None,
    books_file: Optional[Path] = None,
    main_dataset_file: Optional[Path] = None,
    spacy_model: str = "en_core_web_sm",
    min_sentence_length: int = 10,
    max_sentences: Optional[int] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Load raw sentences from reviews without any text cleaning.
    
    This function:
    1. Loads reviews and books datasets
    2. Joins them on work_id
    3. Splits reviews into sentences using spaCy (NO cleaning)
    4. Preserves metadata: work_id, pop_tier, review_id, rating
    
    Args:
        reviews_file: Path to reviews CSV. If None, uses default.
        books_file: Path to books CSV. If None, uses default.
        main_dataset_file: Path to main dataset for book_id->work_id mapping.
        spacy_model: spaCy model name for sentence splitting.
        min_sentence_length: Minimum character length for sentences.
        max_sentences: Maximum number of sentences to return (for testing). If None, returns all.
        seed: Random seed if max_sentences is used.
    
    Returns:
        DataFrame with columns:
        - sentence_id: Unique identifier
        - sentence_text: Raw sentence text (UNPREPROCESSED)
        - review_id: Source review ID
        - work_id: Book/work ID
        - pop_tier: Quality tier
        - rating: Review rating
        - sentence_index: Position within review
    """
    logger.info("=" * 80)
    logger.info("Loading Raw Sentences from Reviews")
    logger.info("=" * 80)
    
    # Import data loading functions
    try:
        from ..data_loading import load_joined_reviews
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from reviews_analysis.data_loading import load_joined_reviews
    
    # Load joined reviews
    logger.info("Loading joined reviews and books...")
    joined_df = load_joined_reviews(
        books_file=books_file,
        reviews_file=reviews_file,
        main_dataset_file=main_dataset_file
    )
    logger.info(f"  Loaded {len(joined_df):,} reviews")
    
    # Load spaCy model (only for sentence splitting)
    logger.info(f"Loading spaCy model: {spacy_model}")
    try:
        nlp = spacy.load(spacy_model, disable=['tagger', 'parser', 'ner', 'lemmatizer', 'attribute_ruler'])
        if 'sentencizer' not in nlp.pipe_names:
            nlp.add_pipe('sentencizer')
        logger.info("  ✓ Model loaded")
    except OSError:
        logger.error(f"spaCy model '{spacy_model}' not found. Install with: python -m spacy download {spacy_model}")
        raise
    
    # Prepare review texts
    logger.info("Preparing review texts...")
    review_texts = []
    metadata = []
    
    for idx, row in joined_df.iterrows():
        text = row['review_text']
        if pd.isna(text) or not isinstance(text, str) or len(str(text).strip()) == 0:
            continue
        
        review_texts.append(str(text))
        metadata.append({
            'review_id': row['review_id'],
            'work_id': row['work_id'],
            'pop_tier': row['pop_tier'],
            'rating': row.get('rating', None)
        })
    
    logger.info(f"  Prepared {len(review_texts):,} reviews for sentence splitting")
    
    # Split reviews into sentences (NO CLEANING)
    logger.info("Splitting reviews into sentences (raw, unpreprocessed)...")
    sentence_records = []
    sentence_id = 0
    
    for review_idx, (text, meta) in enumerate(tqdm(zip(review_texts, metadata), total=len(review_texts), desc="Splitting reviews")):
        # Process with spaCy
        doc = nlp(text)
        
        # Extract sentences (raw, no cleaning)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) >= min_sentence_length]
        
        # Create records for each sentence
        for sent_idx, sentence_text in enumerate(sentences):
            sentence_records.append({
                'sentence_id': sentence_id,
                'sentence_text': sentence_text,  # RAW, UNPREPROCESSED
                'review_id': meta['review_id'],
                'work_id': meta['work_id'],
                'pop_tier': meta['pop_tier'],
                'rating': meta['rating'],
                'sentence_index': sent_idx,
                'n_sentences_in_review': len(sentences)
            })
            sentence_id += 1
    
    # Create DataFrame
    df = pd.DataFrame(sentence_records)
    logger.info(f"  Created {len(df):,} raw sentences")
    
    # Sample if max_sentences specified
    if max_sentences and len(df) > max_sentences:
        logger.info(f"  Sampling {max_sentences:,} sentences (random seed={seed})")
        df = df.sample(n=max_sentences, random_state=seed).reset_index(drop=True)
        df['sentence_id'] = range(len(df))
        logger.info(f"  Final dataset: {len(df):,} sentences")
    
    # Show distribution
    if 'pop_tier' in df.columns:
        logger.info(f"\nPop tier distribution:")
        tier_counts = df['pop_tier'].value_counts()
        for tier, count in tier_counts.items():
            pct = count / len(df) * 100
            logger.info(f"  {tier}: {count:,} ({pct:.1f}%)")
    
    logger.info(f"  Unique reviews: {df['review_id'].nunique():,}")
    logger.info(f"  Unique works: {df['work_id'].nunique():,}")
    
    return df


def load_raw_sentences_from_parquet(
    parquet_file: Path,
    max_sentences: Optional[int] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Load sentences from parquet file.
    
    NOTE: This assumes the parquet has cleaned text. For production use,
    use load_raw_sentences_from_reviews() to get truly raw text.
    
    Args:
        parquet_file: Path to parquet file
        max_sentences: Maximum sentences to load (for testing)
        seed: Random seed if sampling
    
    Returns:
        DataFrame with sentences
    """
    logger.info(f"Loading sentences from parquet: {parquet_file}")
    df = pd.read_parquet(parquet_file)
    logger.info(f"  Loaded {len(df):,} sentences")
    
    if max_sentences and len(df) > max_sentences:
        logger.info(f"  Sampling {max_sentences:,} sentences")
        df = df.sample(n=max_sentences, random_state=seed).reset_index(drop=True)
        df['sentence_id'] = range(len(df))
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load raw sentences from reviews")
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_INTERIM / "review_sentences_raw.parquet",
        help="Output parquet file"
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="Maximum sentences to extract (for testing)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    df = load_raw_sentences_from_reviews(
        max_sentences=args.max_sentences,
        seed=args.seed
    )
    
    logger.info(f"\nSaving to: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    logger.info(f"  ✓ Saved {len(df):,} sentences")

