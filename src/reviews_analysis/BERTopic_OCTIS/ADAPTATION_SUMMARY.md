# BERTopic+OCTIS Adaptation Summary

## ✅ Completed Adaptations

### 1. Data Loading ✅
- **Changed from**: CSV reading with `Author`, `Book Title`, `Chapter`, `Sentence` columns
- **Changed to**: Loads raw (unpreprocessed) sentences from reviews using `load_raw_sentences_from_reviews()`
- **Key feature**: Uses **raw, unpreprocessed text** for embeddings (critical for sentence transformers)
- **Metadata preserved**: `work_id`, `pop_tier`, `review_id`, `rating`

### 2. OCTIS Format Conversion ✅
- **Changed from**: Labels as `"author,book_title"`
- **Changed to**: Labels as `work_id` (book-level topic analysis)
- **Preserved**: `pop_tier` in metadata for correlation analysis with book ratings

### 3. Text Cleaning ✅
- **Removed**: All text cleaning before embeddings
- **Reason**: Sentence transformer models need raw, unpreprocessed text
- **Note**: Text cleaning can be done post-processing in BERTopic if needed

### 4. GPU/CPU Fallback ✅
- **Added**: Automatic detection of GPU availability
- **UMAP/HDBSCAN**: Falls back to CPU versions if cuML not available
- **Embeddings**: Uses GPU if available, falls back to CPU
- **Logging**: Clear messages about which device is being used

### 5. Embedding Model ✅
- **Changed from**: Multiple models (`all-MiniLM-L12-v2`, `multi-qa-mpnet-base-cos-v1`)
- **Changed to**: Single model `all-mpnet-base-v2` for initial testing
- **Device**: Automatically uses GPU if available

### 6. Path Configuration ✅
- **Changed from**: Hardcoded paths (`./data/...`)
- **Changed to**: Project-relative paths using `Path` objects
- **Paths**:
  - Input: `data/processed/review_sentences_for_bertopic.parquet`
  - OCTIS: `data/interim/octis_reviews/`
  - Results: `data/interim/octis_reviews/optimization_results/`
  - Embeddings: `data/interim/octis_reviews/embeddings_*.pkl`

### 7. Email Notifications ✅
- **Removed**: All email notification code
- **Replaced with**: Proper logging using Python `logging` module

### 8. Logging ✅
- **Added**: Comprehensive logging throughout
- **Features**:
  - Progress tracking for large datasets
  - GPU/CPU status messages
  - Error reporting with full tracebacks
  - Summary statistics

## 📁 New Files Created

1. **`sample_test_dataset.py`** - Script to sample test datasets from main parquet
2. **`load_raw_sentences.py`** - Function to load raw (unpreprocessed) sentences from reviews
3. **`ADAPTATION_PLAN.md`** - Detailed adaptation plan
4. **`ADAPTATION_SUMMARY.md`** - This file

## 🔧 Key Technical Changes

### Raw Text for Embeddings
**Critical**: The script now uses **raw, unpreprocessed text** for sentence transformer embeddings. This is important because:
- Transformer models are trained on raw text
- Preprocessing (lowercasing, normalization) can hurt embedding quality
- BERTopic can handle text cleaning internally if needed

### Data Flow
```
Raw Reviews (review_text)
    ↓
Split into sentences (spaCy, NO cleaning)
    ↓
Extract raw sentence_text
    ↓
Calculate embeddings (SentenceTransformer on raw text)
    ↓
BERTopic topic modeling
    ↓
OCTIS optimization
```

### Label Format
- **OCTIS labels**: `work_id` (book-level grouping)
- **Metadata preserved**: `pop_tier` for correlation analysis
- **Future use**: Can analyze topic distribution by `pop_tier`

## 🚀 Usage

### 1. Create Test Dataset (Recommended First Step)
```bash
cd /home/polina/Documents/goodreads_romance_research_cursor/romance_novel_nlp_research

# Activate venv
source romance-novel-nlp-research/.venv/bin/activate

# Sample 10K sentences for testing
python src/reviews_analysis/BERTopic_OCTIS/sample_test_dataset.py \
    --n_samples 10000 \
    --output data/interim/review_sentences_test_10k.parquet
```

### 2. Run BERTopic+OCTIS Optimization
```bash
# Using the test dataset first
python src/reviews_analysis/BERTopic_OCTIS/bertopic_plus_octis.py
```

### 3. Monitor Progress
The script now includes comprehensive logging. Watch for:
- Dataset loading progress
- Embedding calculation progress
- Optimization iterations
- GPU/CPU usage messages

## ⚙️ Configuration

### Virtual Environment
All scripts should be run in:
```
/home/polina/Documents/goodreads_romance_research_cursor/romance_novel_nlp_research/romance-novel-nlp-research/.venv
```

### Embedding Model
Currently set to: `all-mpnet-base-v2`
- Can be changed in `bertopic_plus_octis.py` line ~441

### Test Dataset Size
For initial testing, modify `load_raw_sentences_from_reviews()` call:
```python
df = load_raw_sentences_from_reviews(
    max_sentences=10000,  # Test with 10K sentences
    seed=42
)
```

## 📊 Expected Outputs

1. **OCTIS Corpus**: `data/interim/octis_reviews/corpus.tsv`
2. **Embeddings**: `data/interim/octis_reviews/embeddings_all-mpnet-base-v2.pkl`
3. **Optimization Results**: `data/interim/octis_reviews/optimization_results/all-mpnet-base-v2/result.json`

## ⚠️ Important Notes

1. **Raw Text**: The script uses raw, unpreprocessed text for embeddings. This is correct for sentence transformers.

2. **Large Dataset**: The full dataset has 8.6M sentences. Start with a small test dataset (10K-50K) first.

3. **Memory**: Embedding calculation for large datasets requires significant memory. Monitor memory usage.

4. **GPU**: GPU acceleration is used when available. Falls back to CPU automatically.

5. **Pop Tier**: `pop_tier` is preserved in metadata for correlation analysis with book ratings (future work).

## 🔄 Next Steps

1. **Test with small dataset** (10K sentences)
2. **Verify embeddings calculation** works correctly
3. **Test OCTIS format** generation
4. **Run optimization** on test dataset
5. **Scale up** to larger datasets once verified

## 📝 Files Modified

- `bertopic_plus_octis.py` - Main script (adapted)
- `optimizer.py` - No changes needed (works as-is)
- `topic_npy_to_json.py` - No changes needed
- `restart_script.py` - May need path updates

## 🐛 Known Issues / TODO

- [ ] Test with actual data to verify all imports work
- [ ] Verify GPU/CPU fallback works correctly
- [ ] Test embedding calculation with small dataset
- [ ] Verify OCTIS format is correct
- [ ] Test optimization loop

