# BERTopic+OCTIS Adaptation Plan for Reviews Corpus

## Overview

This document outlines the plan to adapt the existing BERTopic+OCTIS pipeline from novel sentences to reader reviews corpus. The reviews dataset contains **8.6 million sentences** from **965,418 reviews** across **5,998 books**.

## Current Code Structure

### Files in `BERTopic_OCTIS/`:
1. **`bertopic_plus_octis.py`** (664 lines) - Main optimization script
2. **`optimizer.py`** (578 lines) - Bayesian Optimization class
3. **`topic_npy_to_json.py`** (49 lines) - Utility to convert topic arrays to JSON
4. **`restart_script.py`** (69 lines) - Auto-restart script for crashes

## Current Data Format (Novels)

**Input**: CSV file with columns:
- `Author` - Author name
- `Book Title` - Book title
- `Chapter` - Chapter identifier
- `Sentence` - Sentence text

**OCTIS Format**: TSV with columns:
- Sentence text
- Partition (always "train")
- Label (format: `"author,book_title"`)

## Target Data Format (Reviews)

**Input**: Parquet file (`data/processed/review_sentences_for_bertopic.parquet`) with columns:
- `sentence_id` - Unique sentence identifier
- `sentence_text` - The sentence text (cleaned, lowercase)
- `review_id` - Source review ID
- `work_id` - Book/work ID
- `pop_tier` - Quality tier (thrash/mid/top)
- `rating` - Review rating (if available)
- `sentence_index` - Position within review
- `n_sentences_in_review` - Total sentences in review

**Dataset Size**: 8,671,667 sentences (508 MB parquet file)

## Required Adaptations

### 1. Data Loading (HIGH PRIORITY)

**Current Code** (lines 157-205 in `bertopic_plus_octis.py`):
```python
# Reads CSV with latin1 encoding
with open(dataset_path, 'r', encoding='latin1', errors='ignore') as file:
    reader = csv.reader(file, ...)
    # Expects 4 columns: author, book_title, chapter, sentence
    for row in reader:
        if len(row) == 4:
            all_rows.append(row)
df = pd.DataFrame(all_rows, columns=headers)
df['Sentence'] = df['Sentence'].apply(...)  # Text cleaning
dataset_as_list_of_strings = df['Sentence'].tolist()
```

**Required Changes**:
- Replace CSV reading with Parquet reading
- Use `sentence_text` column instead of `Sentence`
- Handle larger dataset (8.6M rows) - may need chunking for memory
- Remove text cleaning (already done in preparation stage)
- Preserve metadata columns for later use

**New Code Structure**:
```python
import pandas as pd
from pathlib import Path

# Load parquet file
dataset_path = Path("data/processed/review_sentences_for_bertopic.parquet")
df = pd.read_parquet(dataset_path)

# Extract sentences (already cleaned)
dataset_as_list_of_strings = df['sentence_text'].tolist()

# Store metadata for later use
metadata = df[['sentence_id', 'review_id', 'work_id', 'pop_tier', 'rating']].copy()
```

### 2. OCTIS Format Conversion (HIGH PRIORITY)

**Current Code** (lines 207-231):
```python
# Creates TSV with: sentence, partition, label
# Label format: "author,book_title"
for row in csv_reader:
    if len(row) == 4:
        author, book_title, chapter, sentence = row
        partition = 'train'
        label = author + "," + book_title
        tsv_data.append([sentence, partition, label])
```

**Required Changes**:
- Use review metadata for labels
- Options for label format:
  - Option A: `work_id` (book-level grouping)
  - Option B: `review_id` (review-level grouping)
  - Option C: `pop_tier` (quality tier grouping)
  - Option D: `f"{work_id}_{pop_tier}"` (combined grouping)

**Recommended**: Use `work_id` for book-level topic analysis, or `pop_tier` for quality-based analysis.

**New Code Structure**:
```python
# Create OCTIS TSV format
octis_corpus_path = octis_dataset_path / 'corpus.tsv'

tsv_data = []
for _, row in df.iterrows():
    sentence = row['sentence_text']
    partition = 'train'
    # Option: Use work_id as label (book-level)
    label = str(row['work_id'])
    # Alternative: Use pop_tier for quality-based analysis
    # label = row['pop_tier']
    tsv_data.append([sentence, partition, label])

# Write TSV file
with open(octis_corpus_path, 'w', encoding='utf-8') as tsv_file:
    writer = csv.writer(tsv_file, delimiter='\t')
    for row in tsv_data:
        writer.writerow(row)
```

### 3. Path Configuration (MEDIUM PRIORITY)

**Current Code**: Hardcoded paths:
- `dataset_path = "./data/processed_novels_sentences_new.csv"`
- `octis_dataset_path = "./data/octis"`
- `embedding_file = "precalculated_embeddings.pkl"`
- `results_path = './data/octis/optimization_results/'`

**Required Changes**:
- Make paths configurable via:
  - Command-line arguments
  - Configuration file (YAML/JSON)
  - Environment variables
- Use project-relative paths
- Create output directories automatically

**Recommended Approach**: Use `pathlib.Path` with project root detection:
```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_INTERIM = PROJECT_ROOT / "data" / "intermediate"

dataset_path = DATA_PROCESSED / "review_sentences_for_bertopic.parquet"
octis_dataset_path = DATA_INTERIM / "octis_reviews"
```

### 4. Email Notifications (LOW PRIORITY)

**Current Code**: Email notifications on start/error/success (lines 129-147, 154, 649, 663)

**Required Changes**:
- Remove email functionality OR
- Make it optional via configuration
- Replace with proper logging

**Recommended**: Remove email code, use logging instead.

### 5. Text Cleaning (LOW PRIORITY)

**Current Code** (lines 184-185):
```python
df['Sentence'] = df['Sentence'].apply(lambda x: re.sub(r'\n+', ' ', str(x)) if isinstance(x, str) else str(x))
df['Sentence'] = df['Sentence'].apply(lambda x: re.sub(r'\s+', ' ', x).strip().lower())
```

**Required Changes**:
- Remove text cleaning (already done in preparation stage)
- Sentences in parquet are already:
  - Lowercase
  - Whitespace normalized
  - Minimum length filtered (10 chars)

### 6. Dataset Size Handling (HIGH PRIORITY)

**Challenge**: 8.6M sentences is much larger than typical novel datasets.

**Required Adaptations**:
- **Memory Management**:
  - Load parquet in chunks if needed
  - Process embeddings in batches
  - Clear intermediate variables
  
- **Test Dataset Creation**:
  - Create sampling script for quick testing
  - Sample sizes: 10K, 50K, 100K, 500K sentences
  - Maintain pop_tier distribution if desired

- **Progress Tracking**:
  - Add verbose logging for large dataset
  - Show progress bars for long operations
  - Log memory usage

**Sampling Script Structure**:
```python
def create_test_dataset(
    input_parquet: Path,
    output_parquet: Path,
    n_samples: int = 10000,
    stratify_by: Optional[str] = 'pop_tier',
    seed: int = 42
) -> pd.DataFrame:
    """Sample sentences for testing."""
    df = pd.read_parquet(input_parquet)
    if stratify_by:
        # Stratified sampling
        sampled = df.groupby(stratify_by).apply(
            lambda x: x.sample(min(len(x), n_samples // 3), random_state=seed)
        ).reset_index(drop=True)
    else:
        sampled = df.sample(n=n_samples, random_state=seed)
    sampled.to_parquet(output_parquet, index=False)
    return sampled
```

### 7. Embedding Model Selection (MEDIUM PRIORITY)

**Current Code** (lines 443-449):
```python
embedding_model_names = [
    "all-MiniLM-L12-v2",
    "multi-qa-mpnet-base-cos-v1"
]
```

**Considerations**:
- Reviews may have different language patterns than novels
- May want to test additional models optimized for reviews/sentiment
- Keep current models for initial testing

**No changes needed initially**, but document for future experimentation.

### 8. Hyperparameter Search Space (LOW PRIORITY)

**Current Code** (lines 522-531):
```python
search_space = {
    'umap__n_neighbors': Integer(2, 50),
    'umap__n_components': Integer(2, 10),
    'umap__min_dist': Real(0.0, 0.1),
    'hdbscan__min_cluster_size': Integer(50, 500),
    'hdbscan__min_samples': Integer(10, 100),
    'vectorizer__min_df': Real(0.001, 0.01),
    'bertopic__top_n_words': Integer(10, 40),
    'bertopic__min_topic_size': Integer(10, 250)
}
```

**Considerations**:
- For 8.6M sentences, may need to adjust:
  - `hdbscan__min_cluster_size`: Larger dataset may need larger min cluster size
  - `vectorizer__min_df`: May need adjustment for review vocabulary
- Start with current values, adjust based on results

**No changes needed initially**, but document for tuning.

### 9. Logging and Monitoring (MEDIUM PRIORITY)

**Current Code**: Basic print statements

**Required Changes**:
- Add proper logging module
- Log to file and console
- Track:
  - Dataset loading progress
  - Embedding calculation progress
  - Optimization iterations
  - Memory usage
  - Time per operation

**Recommended**: Use Python `logging` module with file and console handlers.

### 10. GPU/CPU Fallback (MEDIUM PRIORITY)

**Current Code**: Assumes GPU available (lines 100-101, 456):
```python
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
# ...
embedding_model = SentenceTransformer(embedding_model_name, device="cuda")
```

**Required Changes**:
- Add GPU availability check
- Fallback to CPU versions if GPU unavailable:
  ```python
  try:
      from cuml.cluster import HDBSCAN
      from cuml.manifold import UMAP
      use_gpu = True
  except ImportError:
      from hdbscan import HDBSCAN
      from umap import UMAP
      use_gpu = False
  ```

## Implementation Steps

### Phase 1: Basic Adaptation (Core Functionality)
1. ✅ Review existing code structure
2. ⏳ Adapt data loading: CSV → Parquet
3. ⏳ Update column names: `Sentence` → `sentence_text`
4. ⏳ Adapt OCTIS format conversion for review labels
5. ⏳ Remove text cleaning (already done)
6. ⏳ Test with small sample dataset

### Phase 2: Configuration and Paths
1. ⏳ Make paths configurable
2. ⏳ Remove hardcoded paths
3. ⏳ Add configuration file support
4. ⏳ Create output directories automatically

### Phase 3: Testing and Sampling
1. ⏳ Create test dataset sampling script
2. ⏳ Test with 10K, 50K, 100K samples
3. ⏳ Verify OCTIS format generation
4. ⏳ Test embedding calculation
5. ⏳ Test optimization loop

### Phase 4: Production Readiness
1. ⏳ Add proper logging
2. ⏳ Add GPU/CPU fallback
3. ⏳ Remove email notifications
4. ⏳ Add memory management for large dataset
5. ⏳ Add progress tracking
6. ⏳ Test with full dataset (8.6M sentences)

### Phase 5: Documentation
1. ⏳ Update README with new usage
2. ⏳ Document configuration options
3. ⏳ Document label format choices
4. ⏳ Create example scripts

## File Structure After Adaptation

```
src/reviews_analysis/BERTopic_OCTIS/
├── bertopic_plus_octis.py      # Adapted main script
├── optimizer.py                 # No changes needed
├── topic_npy_to_json.py         # No changes needed
├── restart_script.py            # Update path if needed
├── sample_test_dataset.py       # NEW: Sampling script
├── config.yaml                  # NEW: Configuration file
└── ADAPTATION_PLAN.md           # This file
```

## Key Decisions Needed

1. **Label Format for OCTIS**: 
   - `work_id` (book-level) - Recommended for book topic analysis
   - `pop_tier` (quality-level) - For quality-based topic comparison
   - `review_id` (review-level) - Too granular, not recommended

2. **Test Dataset Size**: Start with 10K-50K for initial testing

3. **Embedding Models**: Keep current models initially, test others later

4. **Hyperparameters**: Start with current values, tune based on results

## Notes

- The reviews dataset is **much larger** (8.6M sentences) than typical novel datasets
- Text is **already cleaned** in the preparation stage
- Metadata is **rich** (review_id, work_id, pop_tier, rating) - can be used for analysis
- Consider **stratified sampling** by pop_tier for balanced test datasets
- **Memory management** is critical for full dataset processing

