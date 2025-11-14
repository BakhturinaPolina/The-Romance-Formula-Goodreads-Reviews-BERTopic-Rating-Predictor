# BERTopic + OCTIS Pipeline - Simple Description

## What is This Pipeline?

This pipeline combines two tools to automatically discover topics in text:
1. **BERTopic** - Finds topics in documents by grouping similar sentences together
2. **OCTIS** - Automatically finds the best settings (hyperparameters) for BERTopic

## How It Works (Simple Overview)

### Step 1: Prepare Data
- Takes sentences from a CSV/Parquet file
- Converts them into a format that OCTIS can understand (TSV format)
- Each row contains: sentence text, partition (usually "train"), and label

### Step 2: Create Embeddings
- Converts each sentence into numbers (called "embeddings")
- This is expensive, so we do it once and save it
- Uses models like "all-MiniLM-L12-v2" or "multi-qa-mpnet-base-cos-v1"

### Step 3: Create Custom BERTopic Model for OCTIS
- Wraps BERTopic in a special adapter class (`BERTopicOctisModelWithEmbeddings`)
- This allows OCTIS to control BERTopic's settings
- The model has many settings that can be tuned:
  - **UMAP**: Reduces dimensions (how many neighbors, how many dimensions)
  - **HDBSCAN**: Groups sentences into topics (minimum cluster size, minimum samples)
  - **Vectorizer**: How to count words (stop words, minimum document frequency)
  - **BERTopic**: General settings (number of words per topic, minimum topic size)

### Step 4: Hyperparameter Optimization
- OCTIS uses **Bayesian Optimization** to find the best settings
- It tries many different combinations of settings
- For each combination, it:
  1. Trains BERTopic with those settings
  2. Measures how good the topics are (using "coherence" - do words in topics make sense together?)
  3. Also measures "diversity" - are topics different from each other?
  4. Saves the results
- After many tries, it finds the best combination

### Step 5: Output
- Saves the best hyperparameters found
- Can retrain models with those best settings
- Saves topic information, topic-word matrices, and topic-document matrices

## Key Components

### Main Script: `bertopic_runner.py`
- Loads data from CSV
- Converts to OCTIS format
- Pre-calculates embeddings
- Runs optimization for multiple embedding models
- Saves results

### Retrain Script: `retrain_from_tables.py`
- Takes optimized hyperparameters (from tables/results)
- Retrains BERTopic models with those exact settings
- Saves final models and topic information

### Optimizer: `hparam_search.py` (in stage04_experiments)
- Contains the `Optimizer` class
- Handles Bayesian Optimization logic
- Manages search space, evaluation, and result saving

### Common Utilities: `src/common/`
- `config.py` - Loads YAML configuration files
- `logging.py` - Sets up logging
- `metrics.py` - Calculates topic quality metrics
- `training_utils.py` - GPU checking, output directory setup

## Data Flow

```
CSV/Parquet File
    ↓
Convert to OCTIS TSV format
    ↓
Calculate embeddings (once, save them)
    ↓
For each embedding model:
    ↓
    Create BERTopicOctisModelWithEmbeddings adapter
    ↓
    Run Bayesian Optimization:
        - Try different hyperparameter combinations
        - Train BERTopic for each combination
        - Measure coherence and diversity
        - Save results
    ↓
Save best hyperparameters
    ↓
(Optional) Retrain with best hyperparameters
    ↓
Final topic models and results
```

## Key Concepts

### Embeddings
- Numbers that represent the meaning of text
- Similar sentences have similar embeddings
- Calculated using pre-trained models (SentenceTransformers)

### UMAP
- Reduces the number of dimensions in embeddings
- Makes it easier to group similar sentences
- Settings: n_neighbors, n_components, min_dist

### HDBSCAN
- Groups sentences into topics (clusters)
- Settings: min_cluster_size, min_samples
- Sentences that don't fit any topic are marked as "outliers" (-1)

### Coherence
- Measures if words in a topic make sense together
- Higher is better
- Example: Topic with words ["love", "romance", "heart"] has high coherence
- Topic with words ["car", "romance", "computer"] has low coherence

### Diversity
- Measures if topics are different from each other
- Higher is better
- If all topics have similar words, diversity is low

### Bayesian Optimization
- Smart way to search for best settings
- Learns from previous tries
- Focuses on promising areas of the search space
- More efficient than trying all combinations randomly

## Current Project Structure (romantic_novels_project_code)

```
src/
├── stage03_modeling/
│   ├── bertopic_runner.py      # Main optimization script
│   ├── retrain_from_tables.py  # Retrain with best params
│   └── main.py                  # CLI entry point
├── stage04_experiments/
│   └── hparam_search.py         # Optimizer class
└── common/
    ├── config.py                # Config loading
    ├── logging.py               # Logging setup
    ├── metrics.py               # Topic metrics
    └── training_utils.py        # GPU/utils
```

## What Needs to Be Adapted

For the reviews dataset, we need to:
1. **Change data loading** - Load from parquet instead of CSV
2. **Adapt column names** - Use `sentence_text` instead of `Sentence`
3. **Update OCTIS format conversion** - Map review metadata (review_id, work_id, pop_tier) to labels
4. **Adjust for larger dataset** - The reviews dataset is much larger (8.6M sentences vs smaller novel dataset)
5. **Create test sampling** - Sample smaller datasets for quick testing

