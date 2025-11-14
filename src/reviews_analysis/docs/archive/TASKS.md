# Task Breakdown: Review-Based Topic Modeling Pipeline

Generated from `plan.md` and `background.md`.

## Status
- ✅ Spec files created (`plan.md`, `background.md`)
- ⚠️ MCP server `task-researcher` is configured but tools may need Cursor restart
- 📋 Manual task breakdown below (can be synced with MCP server once connection is verified)

---

## Phase 1: Data Loading and Linking

### Task 1.1: Create data loading module
**File**: `src/reviews_analysis/data_loading.py`

**Functions to implement**:
- `load_books()` → Load `data/processed/romance_subdataset_6000.csv`
  - Returns: DataFrame with columns including `work_id`, `pop_tier`
- `load_reviews()` → Load `data/processed/romance_reviews_english.csv`
  - Returns: DataFrame with columns including `book_id`, `review_text`, `rating`
- `load_joined_reviews()` → Join reviews to books
  - Join key: `book_id` (reviews) ↔ `work_id` (books)
  - Returns: Joined DataFrame with all book and review columns
- Schema validation: Check ID types, document any mismatches

**Dependencies**: None (first module)

**Outputs**:
- Modular data loading functions
- Documentation of join keys and ID mappings

---

## Phase 2: Coverage Verification

### Task 2.1: Create coverage checking module
**File**: `src/reviews_analysis/checks_coverage.py`

**Functions to implement**:
- `compute_review_counts(books_df, reviews_df)` → Count reviews per book
  - For each book in books dataset, compute:
    - `n_reviews_english`: Count of English reviews
    - Optionally `n_reviews_total` if accessible from logs
- `generate_coverage_table(books_df, review_counts)` → Create coverage table
  - Join counts back to book dataset, preserving `pop_tier`
  - Add columns: `n_reviews_english`, `has_reviews` (bool)
- `summarize_coverage(coverage_df)` → Generate summary statistics
  - Books with 0 reviews
  - Distribution of review counts overall and by `pop_tier`
  - Print/log summary report
- `save_coverage_table(coverage_df, output_path)` → Save to CSV
  - Output: `data/interim/review_coverage_by_book.csv`

**Dependencies**: Phase 1 (data_loading.py)

**Outputs**:
- Coverage table CSV
- Summary statistics (printed/logged)

---

## Phase 3: Exploratory Data Analysis

### Task 3.1: Create EDA notebook/script
**File**: `src/reviews_analysis/eda_reviews.ipynb` (or `.py`)

**Analysis to perform**:
1. **Review count distributions**:
   - Overall distribution of reviews per book
   - Distribution by `pop_tier` (trash/middle/top)
   - Visualizations: histograms, boxplots

2. **Review length analysis**:
   - Character count per review
   - Token count per review (if tokenization available)
   - Distribution by `pop_tier`
   - Visualizations: histograms, violin plots

3. **Ratings analysis** (if available):
   - Overall ratings distribution
   - Ratings by `pop_tier`
   - Correlation between ratings and review length
   - Visualizations: histograms, boxplots

4. **Lexical analysis**:
   - Frequent words/phrases per `pop_tier` group
   - After cleaning and stopword removal
   - Word clouds or frequency bar charts

**Dependencies**: Phase 1, Phase 2

**Outputs**:
- EDA notebook with all analyses
- Figures saved to `reports/reviews_eda/`
- Summary insights documented

---

## Phase 4: Prepare BERTopic Input

### Task 4.1: Create document preparation module
**File**: `src/reviews_analysis/prepare_bertopic_input.py`

**Functions to implement**:
- `aggregate_reviews_per_book(joined_df)` → Aggregate reviews
  - Group by `book_id` (or `work_id`)
  - Concatenate all `review_text` for each book
  - Optionally cap total length (e.g., max 10,000 tokens per book)
- `create_bertopic_documents(aggregated_df)` → Format for BERTopic
  - Output columns:
    - `book_id` (or `work_id`)
    - `pop_tier` (trash/middle/top)
    - `concat_reviews_text` (aggregated review text)
    - `n_reviews` (count of reviews aggregated)
    - `total_length` (character/token count)
- `save_bertopic_input(documents_df, output_path)` → Save dataset
  - Output: `data/processed/review_documents_per_book.csv` (or `.parquet`)

**Dependencies**: Phase 1, Phase 2

**Outputs**:
- Per-book review documents dataset
- Documentation of aggregation strategy

---

## Phase 5: Adapt BERTopic + OCTIS for Reviews

### Task 5.1: Study existing BERTopic code
**Location**: `/home/polina/Documents/goodreads_romance_research_cursor/romantic_novels_project_code/src/stage03_modeling`

**Key files to review**:
- `bertopic_runner.py`: Main BERTopic wrapper with OCTIS integration
- `retrain_from_tables.py`: Model retraining logic
- Understand:
  - Embedding models used
  - UMAP/HDBSCAN hyperparameters
  - Vectorizer configurations (CountVectorizer, ClassTfidfTransformer)
  - OCTIS integration and metrics

**Dependencies**: None (research task)

**Outputs**:
- Notes on existing hyperparameters
- List of adaptations needed for reviews

---

### Task 5.2: Create review-based BERTopic runner
**File**: `src/reviews_analysis/run_bertopic_reviews.py`

**Functions to implement**:
- `load_review_documents(input_path)` → Load prepared documents
- `prepare_embeddings(documents, model_name)` → Generate embeddings
  - Consider models better for short texts (e.g., `all-MiniLM-L6-v2`)
- `train_bertopic_model(documents, embeddings, config)` → Train model
  - Adjust hyperparameters:
    - `min_topic_size`: Likely smaller than 127 (e.g., 10-30 for reviews)
    - UMAP `n_neighbors`: Adjust for shorter documents
    - HDBSCAN `min_cluster_size`: Adjust accordingly
    - Vectorizer `min_df`: May need adjustment for shorter texts
- `evaluate_with_octis(model, documents, topics)` → Compute metrics
  - Topic coherence (C_V)
  - Topic diversity
- `save_model_artifacts(model, output_dir)` → Save results
  - Model file
  - Topic-word lists
  - Topic-document assignments
  - OCTIS metrics JSON

**Dependencies**: Phase 4, Task 5.1

**Outputs**:
- Trained BERTopic model
- Topic assignments and word lists
- OCTIS evaluation metrics

---

### Task 5.3: Create configuration directory
**Directory**: `src/reviews_analysis/config/`

**Files to create**:
- `bertopic_reviews.yaml` (or `.json`): Hyperparameters for review-based modeling
  - Embedding model selection
  - UMAP parameters
  - HDBSCAN parameters
  - Vectorizer settings
  - BERTopic settings (min_topic_size, top_n_words, etc.)
- Optionally: Separate configs for global vs. per-group models

**Dependencies**: Task 5.1

**Outputs**:
- Configuration files for review-based BERTopic

---

### Task 5.4 (Optional): Per-group models
**Extension**: Train separate BERTopic models for each `pop_tier` group

**Functions to add**:
- `train_per_group_models(documents_df, config)` → Train 3 models
  - One for `trash`
  - One for `middle`
  - One for `top`
- Compare topics across groups

**Dependencies**: Task 5.2

**Outputs**:
- Three additional BERTopic models
- Comparative analysis of topics by quality group

---

## Phase 6: Documentation

### Task 6.1: Create README
**File**: `src/reviews_analysis/README.md`

**Sections to include**:
1. **Pipeline Overview**: High-level description of the review-based topic modeling pipeline
2. **Execution Guide**: Step-by-step instructions
   - Which scripts/notebooks to run
   - In what order
   - Required inputs and expected outputs
3. **Key Decisions**:
   - Minimum review count per book (if any filtering applied)
   - Text cleaning rules
   - Aggregation strategy (how reviews combined per book)
   - Hyperparameter adjustments for reviews vs. novels
4. **Output Locations**: Where to find results
   - Coverage tables
   - EDA figures
   - Prepared documents
   - Model artifacts
   - Evaluation metrics

**Dependencies**: All previous phases

**Outputs**:
- Complete README with pipeline documentation

---

## Implementation Order

1. ✅ **Phase 1**: Data loading (foundation)
2. ✅ **Phase 2**: Coverage checks (validation)
3. ✅ **Phase 3**: EDA (understanding)
4. ✅ **Phase 4**: Document preparation (preprocessing)
5. ✅ **Phase 5**: BERTopic modeling (core analysis)
6. ✅ **Phase 6**: Documentation (finalization)

---

## Notes on MCP Server

The `task-researcher` MCP server is configured in `~/.cursor/mcp.json`:
```json
"task-researcher": {
  "command": "/home/polina/.local/bin/task-researcher",
  "args": ["serve-mcp"]
}
```

**If MCP tools become available**, use:
- `parse_inputs` → Parse `plan.md` and `background.md` into tasks
- `analyze_complexity` → Assess task complexity
- `expand_task` / `expand_all_tasks` → Break tasks into subtasks with research
- `update_tasks` → Sync task list as implementation progresses
- `validate_dependencies` / `fix_dependencies` → Maintain task graph
- `generate_task_files` → Export task summaries

**To fix MCP connection**:
1. Restart Cursor to refresh MCP server connections
2. Verify the binary exists: `/home/polina/.local/bin/task-researcher`
3. Check Cursor's MCP logs for connection errors

