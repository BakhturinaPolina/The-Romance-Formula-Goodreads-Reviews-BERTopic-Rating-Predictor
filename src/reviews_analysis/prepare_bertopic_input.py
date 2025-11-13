"""
Prepare sentence-level dataset for BERTopic topic modeling on reviews.

This module splits reviews into sentences and creates a dataset where each row
represents a single sentence with metadata linking it back to its review and book.
This follows the same approach as the original novels pipeline where sentences
were the unit of analysis for BERTopic.
"""

import pandas as pd
import numpy as np
import logging
import spacy
import time
from pathlib import Path
from typing import Optional, List, Tuple
from tqdm import tqdm

# Import data loading functions
try:
    from .data_loading import load_joined_reviews
except ImportError:
    # If running as script, use absolute import
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from reviews_analysis.data_loading import load_joined_reviews

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"


def load_spacy_model(model_name: str = "en_core_web_sm", disable_components: Optional[List[str]] = None) -> spacy.Language:
    """
    Load spaCy model for sentence splitting.
    
    Optimized to disable unnecessary pipeline components for faster processing.
    For sentence splitting, we only need the tokenizer and sentence segmenter.
    
    Args:
        model_name: Name of spaCy model to load. Default is 'en_core_web_sm'.
        disable_components: List of pipeline components to disable.
                           Default: ['tagger', 'parser', 'ner', 'lemmatizer', 'attribute_ruler']
                           (keeps only tokenizer and sentence segmenter)
    
    Returns:
        Loaded spaCy language model with optimized pipeline.
    
    Raises:
        OSError: If the model is not installed.
    """
    if disable_components is None:
        # Disable everything except tokenizer and sentence segmenter for speed
        disable_components = ['tagger', 'parser', 'ner', 'lemmatizer', 'attribute_ruler']
    
    try:
        logger.info(f"Loading spaCy model: {model_name}")
        logger.info(f"Disabling components for speed: {disable_components}")
        nlp = spacy.load(model_name, disable=disable_components)
        logger.info(f"✓ Model loaded successfully")
        logger.info(f"Active pipeline components: {nlp.pipe_names}")
        return nlp
    except OSError:
        logger.error(
            f"spaCy model '{model_name}' not found. "
            f"Install it with: python -m spacy download {model_name}"
        )
        raise


def extract_sentences_from_doc(
    doc,
    min_length: int = 10
) -> List[str]:
    """
    Extract sentences from a spaCy doc object.
    
    Args:
        doc: spaCy Doc object.
        min_length: Minimum character length for a sentence to be included.
    
    Returns:
        List of sentence strings.
    """
    if doc is None:
        return []
    
    # Extract sentences
    sentences = [sent.text.strip() for sent in doc.sents]
    
    # Filter out very short sentences
    sentences = [s for s in sentences if len(s) >= min_length]
    
    return sentences


def split_reviews_to_sentences(
    joined_df: pd.DataFrame,
    nlp: spacy.Language,
    min_sentence_length: int = 10,
    spacy_batch_size: int = 1000,
    log_interval: int = 10000,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Split all reviews into sentences, preserving metadata.
    
    Uses spaCy's batch processing (nlp.pipe) for much faster processing.
    
    Args:
        joined_df: DataFrame with joined reviews and books data.
                  Must contain columns: 'review_id', 'review_text', 'work_id', 'pop_tier'
        nlp: Loaded spaCy language model.
        min_sentence_length: Minimum character length for sentences to include.
        spacy_batch_size: Batch size for spaCy's nlp.pipe() (typically 100-1000).
                         Larger batches = faster but more memory.
        log_interval: Number of reviews to process before logging progress.
        show_progress: Whether to show progress bar.
    
    Returns:
        DataFrame with one row per sentence, containing:
        - sentence_id: Unique identifier for each sentence
        - sentence_text: The sentence text (cleaned)
        - review_id: ID of the source review
        - work_id: ID of the book (work)
        - pop_tier: Quality tier of the book (trash/middle/top)
        - rating: Review rating (if available)
        - sentence_index: Index of sentence within its review (0-based)
        - n_sentences_in_review: Total number of sentences in the source review
    """
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("Splitting reviews into sentences...")
    logger.info(f"Processing {len(joined_df):,} reviews")
    logger.info(f"Using batch processing with batch_size={spacy_batch_size}")
    logger.info(f"Min sentence length: {min_sentence_length}")
    
    # Validate required columns
    logger.debug("Validating required columns...")
    required_cols = ['review_id', 'review_text', 'work_id', 'pop_tier']
    missing_cols = set(required_cols) - set(joined_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    logger.debug(f"✓ All required columns present: {required_cols}")
    
    # Prepare metadata lists (parallel to review_texts)
    logger.debug("Preparing metadata lists...")
    prep_start = time.time()
    review_ids = joined_df['review_id'].tolist()
    work_ids = joined_df['work_id'].tolist()
    pop_tiers = joined_df['pop_tier'].tolist()
    ratings = joined_df['rating'].tolist() if 'rating' in joined_df.columns else [None] * len(joined_df)
    logger.debug(f"✓ Metadata lists prepared in {time.time() - prep_start:.2f}s")
    logger.debug(f"  - Review IDs: {len(review_ids):,}")
    logger.debug(f"  - Work IDs: {len(work_ids):,}")
    logger.debug(f"  - Pop tiers: {len(pop_tiers):,}")
    logger.debug(f"  - Ratings: {len(ratings):,}")
    
    # Prepare review texts (handle NaN and non-string values)
    logger.debug("Preparing review texts and filtering empty reviews...")
    text_prep_start = time.time()
    review_texts = []
    valid_indices = []
    empty_count = 0
    for idx, text in enumerate(joined_df['review_text']):
        if pd.isna(text) or not isinstance(text, str) or len(str(text).strip()) == 0:
            empty_count += 1
            continue
        review_texts.append(str(text))
        valid_indices.append(idx)
    text_prep_time = time.time() - text_prep_start
    logger.info(f"✓ Text preparation complete in {text_prep_time:.2f}s")
    logger.info(f"  Valid reviews to process: {len(review_texts):,}")
    logger.info(f"  Empty reviews skipped: {empty_count:,} ({empty_count/len(joined_df)*100:.1f}%)")
    if len(review_texts) > 0:
        avg_text_length = sum(len(t) for t in review_texts) / len(review_texts)
        logger.debug(f"  Average review text length: {avg_text_length:.0f} characters")
    
    # Prepare list to store sentence records
    sentence_records = []
    
    # Process reviews in batches using spaCy's pipe (much faster!)
    # Process entire batches at once for better performance
    total_processed = 0
    total_sentences = 0
    
    # Create progress bar
    pbar = tqdm(total=len(review_texts), desc="Splitting reviews", 
                unit="review", unit_scale=True) if show_progress else None
    
    # Process in chunks to manage memory
    chunk_size = 50000  # Process 50k reviews at a time
    n_chunks = (len(review_texts) + chunk_size - 1) // chunk_size
    logger.info(f"Processing in {n_chunks} chunks of up to {chunk_size:,} reviews each")
    
    chunk_processing_times = []
    
    for chunk_num, chunk_start in enumerate(range(0, len(review_texts), chunk_size), 1):
        chunk_start_time = time.time()
        chunk_end = min(chunk_start + chunk_size, len(review_texts))
        chunk_texts = review_texts[chunk_start:chunk_end]
        chunk_indices = valid_indices[chunk_start:chunk_end]
        chunk_size_actual = len(chunk_texts)
        
        logger.info(f"[Chunk {chunk_num}/{n_chunks}] Processing reviews {chunk_start:,}-{chunk_end:,} ({chunk_size_actual:,} reviews)")
        
        # Process this chunk with spaCy pipe (larger batch size for efficiency)
        # Use larger batch size since we're processing in chunks
        effective_batch_size = min(spacy_batch_size * 2, 2000)  # Up to 2000 for speed
        logger.debug(f"  Using effective batch size: {effective_batch_size}")
        
        spacy_start = time.time()
        docs = list(nlp.pipe(chunk_texts, batch_size=effective_batch_size, n_process=1))
        spacy_time = time.time() - spacy_start
        logger.debug(f"  ✓ spaCy processing completed in {spacy_time:.2f}s ({chunk_size_actual/spacy_time:.0f} reviews/sec)")
        
        # Process each doc in the chunk
        extraction_start = time.time()
        chunk_sentences = 0
        for local_idx, doc in enumerate(docs):
            orig_idx = chunk_indices[local_idx]
            
            # Get corresponding metadata for this review
            review_id = review_ids[orig_idx]
            work_id = work_ids[orig_idx]
            pop_tier = pop_tiers[orig_idx]
            rating = ratings[orig_idx]
            
            # Extract sentences from doc
            sentences = extract_sentences_from_doc(doc, min_length=min_sentence_length)
            
            # Create one record per sentence
            n_sentences = len(sentences)
            chunk_sentences += n_sentences
            for sent_idx, sentence_text in enumerate(sentences):
                sentence_records.append({
                    'sentence_text': sentence_text,
                    'review_id': review_id,
                    'work_id': work_id,
                    'pop_tier': pop_tier,
                    'rating': rating,
                    'sentence_index': sent_idx,
                    'n_sentences_in_review': n_sentences
                })
            
            total_processed += 1
            total_sentences += n_sentences
            
            # Update progress bar
            if pbar:
                pbar.update(1)
            
            # Log progress periodically
            if total_processed % log_interval == 0:
                elapsed = time.time() - start_time
                rate = total_processed / elapsed if elapsed > 0 else 0
                eta_seconds = (len(review_texts) - total_processed) / rate if rate > 0 else 0
                eta_minutes = eta_seconds / 60
                logger.info(
                    f"Progress: {total_processed:,}/{len(review_texts):,} reviews "
                    f"({total_processed/len(review_texts)*100:.1f}%), "
                    f"{total_sentences:,} sentences extracted, "
                    f"rate: {rate:.0f} reviews/sec, "
                    f"ETA: {eta_minutes:.1f} minutes"
                )
        
        extraction_time = time.time() - extraction_start
        chunk_time = time.time() - chunk_start_time
        chunk_processing_times.append(chunk_time)
        
        logger.info(
            f"  ✓ Chunk {chunk_num} complete: {chunk_size_actual:,} reviews → {chunk_sentences:,} sentences "
            f"in {chunk_time:.2f}s ({chunk_size_actual/chunk_time:.0f} reviews/sec, "
            f"{chunk_sentences/chunk_size_actual:.1f} sentences/review)"
        )
        
        # Log running statistics
        if chunk_num > 0:
            avg_chunk_time = sum(chunk_processing_times) / len(chunk_processing_times)
            remaining_chunks = n_chunks - chunk_num
            estimated_remaining = avg_chunk_time * remaining_chunks
            logger.debug(
                f"  Running average: {avg_chunk_time:.2f}s per chunk, "
                f"estimated {estimated_remaining/60:.1f} minutes remaining"
            )
    
    if pbar:
        pbar.close()
    
    total_processing_time = time.time() - start_time
    logger.info(f"✓ All chunks processed in {total_processing_time:.2f}s ({total_processing_time/60:.1f} minutes)")
    logger.info(f"  Total reviews processed: {total_processed:,}")
    logger.info(f"  Total sentences extracted: {total_sentences:,}")
    logger.info(f"  Average processing rate: {total_processed/total_processing_time:.0f} reviews/sec")
    logger.info(f"  Average sentences per review: {total_sentences/total_processed:.2f}")
    
    # Create DataFrame from records
    logger.info("=" * 80)
    logger.info("Creating DataFrame from sentence records...")
    df_start = time.time()
    logger.debug(f"  Number of sentence records: {len(sentence_records):,}")
    logger.debug(f"  Estimated memory: ~{len(sentence_records) * 500 / 1024 / 1024:.0f} MB")
    
    sentences_df = pd.DataFrame(sentence_records)
    df_time = time.time() - df_start
    logger.info(f"✓ DataFrame created in {df_time:.2f}s")
    logger.debug(f"  DataFrame shape: {sentences_df.shape}")
    logger.debug(f"  DataFrame memory usage: {sentences_df.memory_usage(deep=True).sum() / 1024 / 1024:.0f} MB")
    
    # Add sentence_id as unique identifier
    logger.debug("Adding sentence_id column...")
    sentences_df['sentence_id'] = range(len(sentences_df))
    logger.debug(f"✓ Added sentence_id: {sentences_df['sentence_id'].min()}-{sentences_df['sentence_id'].max()}")
    
    # Reorder columns for clarity
    column_order = [
        'sentence_id',
        'sentence_text',
        'review_id',
        'work_id',
        'pop_tier',
        'rating',
        'sentence_index',
        'n_sentences_in_review'
    ]
    sentences_df = sentences_df[column_order]
    
    # Log summary statistics
    logger.info("=" * 80)
    logger.info("SPLITTING SUMMARY STATISTICS")
    logger.info("=" * 80)
    logger.info(f"✓ Split complete: {len(sentences_df):,} sentences from {total_processed:,} reviews")
    if total_processed > 0:
        logger.info(f"  Average sentences per review: {len(sentences_df) / total_processed:.2f}")
    logger.info(f"  Unique reviews: {sentences_df['review_id'].nunique():,}")
    logger.info(f"  Unique books: {sentences_df['work_id'].nunique():,}")
    
    # Log distribution by pop_tier
    if 'pop_tier' in sentences_df.columns:
        logger.info("  Sentences by pop_tier:")
        tier_counts = sentences_df['pop_tier'].value_counts()
        for tier, count in tier_counts.items():
            pct = count / len(sentences_df) * 100
            logger.info(f"    {tier}: {count:,} ({pct:.1f}%)")
    
    # Log sentence length statistics
    if len(sentences_df) > 0:
        sentence_lengths = sentences_df['sentence_text'].str.len()
        logger.debug("  Sentence length statistics:")
        logger.debug(f"    Min: {sentence_lengths.min()} chars")
        logger.debug(f"    Max: {sentence_lengths.max()} chars")
        logger.debug(f"    Mean: {sentence_lengths.mean():.1f} chars")
        logger.debug(f"    Median: {sentence_lengths.median():.1f} chars")
    
    logger.info("=" * 80)
    
    return sentences_df


def clean_sentence_text(sentences_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean sentence text (similar to original code).
    
    This applies the same cleaning as in the original bertopic_runner.py:
    - Remove extra newlines
    - Normalize whitespace
    - Convert to lowercase
    
    Args:
        sentences_df: DataFrame with 'sentence_text' column.
    
    Returns:
        DataFrame with cleaned 'sentence_text' column.
    """
    logger.info("=" * 80)
    logger.info("Cleaning sentence text...")
    clean_start = time.time()
    initial_count = len(sentences_df)
    logger.debug(f"  Initial sentence count: {initial_count:,}")
    
    sentences_df = sentences_df.copy()
    
    # Apply same cleaning as original code
    logger.debug("  Step 1: Removing extra newlines...")
    step1_start = time.time()
    sentences_df['sentence_text'] = sentences_df['sentence_text'].apply(
        lambda x: ' '.join(str(x).split('\n')) if isinstance(x, str) else str(x)
    )
    logger.debug(f"    ✓ Completed in {time.time() - step1_start:.2f}s")
    
    # Normalize whitespace and convert to lowercase
    logger.debug("  Step 2: Normalizing whitespace and converting to lowercase...")
    step2_start = time.time()
    sentences_df['sentence_text'] = sentences_df['sentence_text'].apply(
        lambda x: ' '.join(str(x).split()).strip().lower() if isinstance(x, str) else str(x)
    )
    logger.debug(f"    ✓ Completed in {time.time() - step2_start:.2f}s")
    
    # Remove empty sentences after cleaning
    logger.debug("  Step 3: Removing empty sentences...")
    step3_start = time.time()
    sentences_df = sentences_df[sentences_df['sentence_text'].str.len() > 0]
    removed = initial_count - len(sentences_df)
    logger.debug(f"    ✓ Completed in {time.time() - step3_start:.2f}s")
    
    clean_time = time.time() - clean_start
    
    if removed > 0:
        logger.warning(f"  Removed {removed:,} empty sentences after cleaning ({removed/initial_count*100:.2f}%)")
    
    logger.info(f"✓ Cleaning complete in {clean_time:.2f}s: {len(sentences_df):,} sentences remaining")
    logger.debug(f"  Cleaning rate: {initial_count/clean_time:.0f} sentences/sec")
    
    return sentences_df


def create_sentence_dataset(
    joined_df: Optional[pd.DataFrame] = None,
    spacy_model: str = "en_core_web_sm",
    min_sentence_length: int = 10,
    clean_text: bool = True,
    spacy_batch_size: int = 1000,
    log_interval: int = 10000
) -> pd.DataFrame:
    """
    Create sentence-level dataset from joined reviews.
    
    This is the main function that orchestrates the sentence extraction process.
    
    Args:
        joined_df: DataFrame with joined reviews and books. If None, loads using
                  data_loading.load_joined_reviews().
        spacy_model: Name of spaCy model to use for sentence splitting.
        min_sentence_length: Minimum character length for sentences.
        clean_text: Whether to clean sentence text (normalize whitespace, lowercase).
        spacy_batch_size: Batch size for spaCy's nlp.pipe() (typically 100-1000).
                         Larger batches = faster but more memory.
        log_interval: Number of reviews to process before logging progress.
    
    Returns:
        DataFrame with sentence-level data ready for BERTopic.
    """
    # Load data if not provided
    if joined_df is None:
        logger.info("Loading joined reviews and books data...")
        joined_df = load_joined_reviews(how="inner")
        logger.info(f"Loaded {len(joined_df):,} reviews")
    
    # Load spaCy model (optimized - only tokenizer and sentence segmenter)
    nlp = load_spacy_model(spacy_model, disable_components=['tagger', 'parser', 'ner', 'lemmatizer', 'attribute_ruler'])
    
    # Split reviews into sentences (using optimized batch processing)
    sentences_df = split_reviews_to_sentences(
        joined_df,
        nlp,
        min_sentence_length=min_sentence_length,
        spacy_batch_size=spacy_batch_size,
        log_interval=log_interval
    )
    
    # Clean text if requested
    if clean_text:
        sentences_df = clean_sentence_text(sentences_df)
    
    return sentences_df


def save_sentence_dataset(
    sentences_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    format: str = "parquet"
) -> Path:
    """
    Save sentence dataset to file.
    
    Args:
        sentences_df: DataFrame with sentence-level data.
        output_path: Path to save the file. If None, uses default location.
        format: File format ('parquet' or 'csv'). Default is 'parquet' for efficiency.
    
    Returns:
        Path to the saved file.
    """
    logger.info("=" * 80)
    logger.info("Saving sentence dataset...")
    save_start = time.time()
    
    if output_path is None:
        output_path = DATA_PROCESSED / f"review_sentences_for_bertopic.{format}"
    
    # Ensure output directory exists
    logger.debug(f"  Output directory: {output_path.parent}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(f"  ✓ Directory exists or created")
    
    logger.info(f"  Saving to: {output_path}")
    logger.debug(f"  Format: {format}")
    logger.debug(f"  Rows to save: {len(sentences_df):,}")
    logger.debug(f"  Columns: {list(sentences_df.columns)}")
    
    # Save based on format
    if format == "parquet":
        logger.debug("  Writing parquet file...")
        write_start = time.time()
        sentences_df.to_parquet(output_path, index=False, engine='pyarrow')
        write_time = time.time() - write_start
        logger.debug(f"    ✓ Parquet write completed in {write_time:.2f}s")
    elif format == "csv":
        logger.debug("  Writing CSV file...")
        write_start = time.time()
        sentences_df.to_csv(output_path, index=False)
        write_time = time.time() - write_start
        logger.debug(f"    ✓ CSV write completed in {write_time:.2f}s")
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'parquet' or 'csv'.")
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    file_size_gb = file_size_mb / 1024
    save_time = time.time() - save_start
    
    logger.info(f"✓ Save complete in {save_time:.2f}s")
    logger.info(f"  File size: {file_size_mb:.2f} MB ({file_size_gb:.3f} GB)")
    logger.info(f"  Write rate: {len(sentences_df)/write_time:.0f} rows/sec")
    logger.debug(f"  File path: {output_path.absolute()}")
    
    return output_path


def main():
    """
    Main function to create and save sentence dataset.
    """
    script_start = time.time()
    logger.info("=" * 80)
    logger.info("PREPARING SENTENCE-LEVEL DATASET FOR BERTOPIC")
    logger.info("=" * 80)
    logger.info(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create sentence dataset with optimized batch processing
    logger.info("\n[STEP 1] Creating sentence dataset...")
    dataset_start = time.time()
    sentences_df = create_sentence_dataset(
        spacy_model="en_core_web_sm",
        min_sentence_length=10,
        clean_text=True,
        spacy_batch_size=2000,  # Larger batch size for faster processing
        log_interval=50000  # Log every 50k reviews
    )
    dataset_time = time.time() - dataset_start
    logger.info(f"✓ Dataset creation completed in {dataset_time/60:.1f} minutes")
    
    # Save dataset
    logger.info("\n[STEP 2] Saving dataset to file...")
    save_start = time.time()
    output_path = save_sentence_dataset(
        sentences_df,
        format="parquet"  # Use parquet for efficiency with large datasets
    )
    save_time = time.time() - save_start
    logger.info(f"✓ Save completed in {save_time:.2f}s")
    
    # Final summary
    total_time = time.time() - script_start
    logger.info("=" * 80)
    logger.info("✓ SENTENCE DATASET PREPARATION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Total execution time: {total_time/60:.1f} minutes ({total_time:.0f} seconds)")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Total sentences: {len(sentences_df):,}")
    logger.info(f"Unique reviews: {sentences_df['review_id'].nunique():,}")
    logger.info(f"Unique books: {sentences_df['work_id'].nunique():,}")
    logger.info(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    # Print sample
    logger.info("\nSample sentences (first 10):")
    sample = sentences_df[['sentence_id', 'sentence_text', 'review_id', 'work_id', 'pop_tier']].head(10)
    for idx, row in sample.iterrows():
        logger.info(f"  [{row['sentence_id']}] {row['sentence_text'][:80]}... (review_id={row['review_id']}, work_id={row['work_id']}, tier={row['pop_tier']})")
    
    return sentences_df, output_path


if __name__ == "__main__":
    sentences_df, output_path = main()

