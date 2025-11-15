# Configuration File Usage Guide

## Overview

The BERTopic+OCTIS pipeline now supports YAML-based configuration for hyperparameter optimization. The configuration file allows you to:

- Specify multiple embedding models to test
- Define hyperparameter search spaces
- Customize optimization parameters

## Configuration File Location

The configuration file should be located at:
```
src/reviews_analysis/BERTopic_OCTIS/config_bertopic_reviews.yaml
```

## Backward Compatibility

If the configuration file is not found, the script will:
- Use default embedding model: `all-mpnet-base-v2`
- Use default hyperparameter search space (as defined in code)
- Log a warning and continue with defaults

## Configuration Structure

### Embedding Models

```yaml
embedding_model:
  type: categorical
  values:
    - "sentence-transformers/paraphrase-mpnet-base-v2"
    - "sentence-transformers/multi-qa-mpnet-base-cos-v1"
    - "intfloat/e5-base-v2"
    - "thenlper/gte-large"
```

The script will:
1. Load all specified embedding models
2. Calculate embeddings for each (or load from cache if exists)
3. Run optimization separately for each model

### Hyperparameters

Each hyperparameter can be defined as:

**Integer (range):**
```yaml
top_n_words:
  type: integer
  min: 8
  max: 25
```

**Categorical (list of values):**
```yaml
n_gram_range:
  type: categorical
  values:
    - [1, 1]
    - [1, 2]
    - [1, 3]
```

**Boolean (as categorical):**
```yaml
low_memory:
  type: categorical
  values:
    - false
    - true
```

## Important Constraints

### HDBSCAN Constraint

The constraint `hdbscan_min_samples ≤ hdbscan_min_cluster_size` is automatically enforced at runtime. If the optimizer suggests a violation, the code will automatically set `min_samples = min_cluster_size`.

### Topic Count Filter

Models with **less than 200 topics** are automatically rejected. This threshold was changed from 100 to 200.

## Verbose Logging

The script now includes extensive logging at each step:

### Step Markers

Each major step is marked with `[STEP N]`:
- `[STEP 1]` - Loading configuration
- `[STEP 2]` - Loading dataset
- `[STEP 3]` - Preparing dataset for embeddings
- `[STEP 4]` - Preparing OCTIS dataset format
- `[STEP 5]` - Loading embedding models
- `[STEP 6]` - Calculating/loading embeddings
- `[STEP 7]` - Setting up hyperparameter optimization
- `[STEP 8]` - Starting optimization for all models

### Training Logs

Each training run logs:
- Hyperparameters being used
- Component initialization (UMAP, HDBSCAN, etc.)
- Model fitting progress
- Topic analysis results
- Memory cleanup

### Debugging Tips

1. **Check configuration loading**: Look for `[STEP 1]` to verify config is loaded
2. **Monitor embedding calculation**: `[STEP 6]` shows progress and timing
3. **Track training runs**: Each training shows detailed hyperparameters and results
4. **Watch for rejections**: Models with <200 topics are logged with rejection reason
5. **Monitor constraints**: Constraint violations are logged and automatically fixed

## Example Output

```
================================================================================
BERTopic + OCTIS Optimization for Reviews Corpus
================================================================================

[STEP 1] Loading configuration...
  Loading configuration from: .../config_bertopic_reviews.yaml
  ✓ Configuration loaded successfully
  Configuration loaded: 12 parameter groups
  Available parameters: embedding_model, top_n_words, n_gram_range, ...

[STEP 2] Loading dataset...
  Checking for test dataset: ...
  ✓ Loaded 10,000 raw sentences
  Columns: ['sentence_text', 'work_id', 'pop_tier', ...]

[STEP 5] Loading embedding models...
  Using 4 embedding models from config:
    1. sentence-transformers/paraphrase-mpnet-base-v2
    2. sentence-transformers/multi-qa-mpnet-base-cos-v1
    ...

[STEP 6] Calculating/loading embeddings...
  [1/4] Processing: sentence-transformers/paraphrase-mpnet-base-v2
    Found existing embeddings file
    ✓ Embeddings loaded in 2.34 seconds
    Embedding shape: (10000, 768)

[STEP 7] Setting up hyperparameter optimization for: sentence-transformers/paraphrase-mpnet-base-v2
  Building search space from configuration...
  Search space contains 11 parameters:
    - bertopic__low_memory
    - bertopic__min_topic_size
    - bertopic__n_gram_range
    ...

================================================================================
Training #1
Embedding model: sentence-transformers/paraphrase-mpnet-base-v2
================================================================================

[Training] Setting hyperparameters...
  Received 11 hyperparameters from optimizer
  Hyperparameters after merging:
    umap:
      n_neighbors: 45
      n_components: 8
      ...
  ✓ Constraint enforced

[Training] Initializing pipeline components...
  Creating UMAP model...
    Parameters: {'n_neighbors': 45, 'n_components': 8, ...}
    Using GPU version (cuML)
    ✓ UMAP model created
  ...

[Training] Fitting BERTopic model...
  Dataset size: 10,000 sentences
  Embeddings shape: (10000, 768)
  ✓ Model fitted in 45.2 seconds (0.75 minutes)

[Training] Analyzing topic assignments...
  Total topic assignments: 10,000
  Number of topics (excluding outliers): 234
  Number of outliers (topic -1): 1,234 (12.3%)
  Checking topic count threshold (minimum: 200)...
  ✓ Model accepted: 234 topics found
```

## Troubleshooting

### Configuration Not Found

If you see:
```
Config file not found: ...
Using default configuration (backward compatibility mode)
```

The script will continue with defaults. To use config:
1. Ensure `config_bertopic_reviews.yaml` exists in the BERTopic_OCTIS directory
2. Check file permissions

### Constraint Violations

If you see:
```
⚠ Constraint violation: min_samples (250) > min_cluster_size (200)
Enforcing constraint: setting min_samples = min_cluster_size = 200
✓ Constraint enforced
```

This is normal - the optimizer may suggest invalid combinations, but they're automatically corrected.

### Model Rejections

If you see:
```
⚠ Model rejected: Only 150 topics found (minimum required: 200)
Rejection reason: Topic count 150 < 200
```

The model didn't produce enough topics. This is logged to the CSV results file.

## Next Steps

1. Customize `config_bertopic_reviews.yaml` with your desired hyperparameters
2. Run the script and monitor the verbose logs
3. Check results in `data/interim/octis_reviews/optimization_results/`
4. Review CSV files for detailed training history

