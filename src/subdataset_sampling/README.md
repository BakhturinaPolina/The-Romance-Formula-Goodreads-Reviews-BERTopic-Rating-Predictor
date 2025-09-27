# Sub-dataset Sampling Module

## Overview

This module implements the research methodology for creating a representative 6,000-book sub-dataset from the main romance novel corpus. The sampling strategy ensures equal representation across popularity tiers while preserving key demographic and engagement characteristics.

## Methodology

### Core Principles

1. **Equal Tier Representation**: Exactly 2,000 books per popularity tier (top, mid, thrash)
2. **Engagement Priority**: Within each tier, prioritize books with richest reader data
3. **Representativeness**: Preserve publication year, genre, series status, and page length distributions
4. **Research-Ready**: Optimized for BERTopic/OCTIS analysis with parameter tuning

### Tier Definition

Books are divided into three popularity tiers based on `average_rating_weighted_mean` quantiles:

- **Thrash**: Bottom 25% (< Q1 ≈ 3.71)
- **Mid**: Middle 50% (Q1 to Q3 ≈ 3.71 to 4.15)
- **Top**: Top 25% (> Q3 ≈ 4.15)

### Engagement Priority

Within each tier, books are ranked by:
1. `ratings_count_sum` (descending)
2. `text_reviews_count_sum` (descending)
3. `author_ratings_count` (descending)

### Representativeness Features

- **Publication Year**: 2000-2017 distribution preserved within each tier
- **Genre Groups**: paranormal, historical, fantasy, mystery, young_adult, other
- **Series Status**: standalone vs. series membership
- **Page Length**: Quartile distribution maintained

## Usage

### Basic Usage

```python
from src.subdataset_sampling import create_subdataset_6000

# Create the sub-dataset
sample_df = create_subdataset_6000(
    input_csv_path="data/processed/romance_books_main_final.csv",
    output_csv_path="data/processed/romance_subdataset_6000.csv"
)
```

### Command Line

```bash
# Using defaults
python src/subdataset_sampling/run_subdataset_sampling.py

# Custom paths
python src/subdataset_sampling/run_subdataset_sampling.py \
    /path/to/input.csv \
    /path/to/output.csv
```

## Output

The sub-dataset includes these key columns for downstream analysis:

- `work_id`: Unique book identifier
- `title`: Book title
- `author_id`, `author_name`: Author information
- `publication_year`: Publication year
- `num_pages_median`: Page count
- `genres_str`: Genre classifications
- `series_id`, `series_title`: Series information
- `ratings_count_sum`: Total ratings across editions
- `text_reviews_count_sum`: Total text reviews
- `average_rating_weighted_mean`: Popularity metric
- `pop_tier`: Assigned popularity tier

## Validation

The sampling process includes comprehensive validation:

1. **Tier Balance**: Confirms exactly 2,000 books per tier
2. **Representativeness**: Compares distributions between full dataset and sample
3. **Engagement Quality**: Reports median engagement metrics by tier

## Research Benefits

This 6,000-book sub-dataset provides:

- **Statistical Power**: Sufficient for BERTopic/OCTIS with parameter tuning
- **Balanced Design**: Equal tiers enable fair theme comparison
- **Clean Splits**: Supports 70/15/15 train/validation/test splits
- **Fast Iteration**: Maintains reasonable size for development
- **Rich Signal**: Prioritizes books with substantial reader engagement data

## Alternative Sizes

The same methodology can create:
- **9,000 books** (3,000 per tier): Larger research dataset
- **3,000 books** (1,000 per tier): Smaller development dataset

Simply modify the `TOTAL` variable in the code.

## Files

- `create_subdataset_6000.py`: Main sampling implementation
- `run_subdataset_sampling.py`: Command-line runner
- `README.md`: This documentation
"""

Now let me run the sampling to create your sub-dataset:

```bash
cd /home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research
python src/subdataset_sampling/run_subdataset_sampling.py
```

## Summary

I've implemented your exact sampling methodology with the following structure:

✅ **Equal Tier Representation**: 2,000 books per tier (top/mid/thrash)  
✅ **Engagement Priority**: Ranked by ratings_count_sum, text_reviews_count_sum, author_ratings_count  
✅ **Representativeness Preserved**: Publication year, genre groups, series status, page length distributions  
✅ **Research-Ready**: 6,000 books optimized for BERTopic/OCTIS analysis  
✅ **Clean Implementation**: Modular code structure with comprehensive validation  

The sampling script will:
1. Load your main dataset
2. Compute tier boundaries from your actual data quantiles
3. Create representative quotas for each tier
4. Sample with engagement priority while preserving demographics
5. Backfill to reach exact counts
6. Export with validation reports

Would you like me to run the sampling now, or would you prefer any modifications to the methodology or implementation?
