# Shelf Normalization Pipeline

**Step 2 of the Romance Novel EDA Analysis Pipeline**

This module implements the shelf normalization pipeline that bridges the audit outputs (Step 1) to semantic shelf clustering (Step 3). It transforms messy shelf tags into normalized, canonical forms suitable for downstream analysis.

## Overview

The shelf normalization pipeline performs three main operations:

1. **Canonicalization (2.1)**: Deterministic normalization of shelf strings
2. **Segmentation (2.2)**: Conservative CamelCase and concatenation splitting
3. **Alias Detection (2.1.B)**: Identification of potential shelf aliases

## Files

- `shelf_normalize.py` — Main pipeline script
- `shelf_filters.yml` — Regex patterns for non-content shelf identification
- `test_shelf_normalize.py` — Comprehensive test suite
- `Makefile` — Easy pipeline execution
- `requirements_audit.txt` — Updated dependencies

## Dependencies

```bash
pip install -r requirements_audit.txt
```

Key dependencies:
- `pandas>=2.0` — Data manipulation
- `pyarrow>=10.0` — Parquet file handling
- `rapidfuzz>=2.0` — String similarity metrics (optional)
- `wordfreq>=3.0` — Word frequency data (optional)
- `pyyaml>=6.0` — YAML configuration

## Usage

### Basic Usage

```bash
# Run with default settings
python shelf_normalize.py \
    --from-audit audit_outputs_full \
    --parsed-parquet parse_outputs_full/parsed_books_*.parquet \
    --filters shelf_filters.yml \
    --outdir shelf_norm_outputs
```

### Using Makefile

```bash
# Run shelf normalization step
make shelf-norm

# Run complete pipeline (audit → parse → shelf-norm)
make pipeline

# Test the pipeline
make test-shelf
```

### Advanced Options

```bash
python shelf_normalize.py \
    --from-audit audit_outputs_full \
    --parsed-parquet parse_outputs_full/parsed_books_*.parquet \
    --filters shelf_filters.yml \
    --outdir shelf_norm_outputs \
    --n-top-shelves 10000 \
    --jw-threshold 0.90 \
    --j3-threshold 0.75 \
    --edit-max 2 \
    --min-seg-evidence 3 \
    --zipf-min 2.5
```

## Input Requirements

### Required Inputs

1. **Audit Directory** (`--from-audit`): Contains:
   - `audit_results.json` — Schema and data quality results
   - `overdispersion_tests.json` — Statistical test results

2. **Parsed Parquet** (`--parsed-parquet`): From `01_parse_lists.py` containing:
   - `shelves` column with list of shelf strings per book

3. **Shelf Filters** (`--filters`): YAML file with regex patterns for non-content shelves

### Optional Inputs

- If `--parsed-parquet` is not provided, the script will attempt to find it in `audit_results.json`

## Output Files

All outputs are written to the specified `--outdir`:

### Core Outputs

1. **`shelf_canonical.csv`** — Canonicalization mapping
   - `shelf_raw`: Original shelf string
   - `shelf_canon`: Canonical form
   - `reason`: Normalization reason
   - `noncontent_category`: Leakage category (if any)
   - `count`: Frequency in dataset

2. **`shelf_segments.csv`** — Segmentation results
   - `shelf_canon`: Canonical shelf
   - `segments`: Space-separated segments
   - `accepted`: Whether segmentation was applied
   - `guard1_standalone`: Guard 1 (standalone evidence)
   - `guard2_lexicon`: Guard 2 (lexicon membership)
   - `evidence_count`: Frequency evidence

3. **`shelf_alias_candidates.csv`** — Potential aliases
   - `shelf_a`, `shelf_b`: Shelf pairs
   - `jw`: Jaro-Winkler similarity
   - `edit`: Edit distance
   - `jaccard3`: 3-gram Jaccard similarity
   - `decision_hint`: Suggested action

4. **`segments_vocab.txt`** — Unique segment vocabulary
   - One segment per line
   - Sorted alphabetically
   - For downstream vectorization

5. **`noncontent_shelves.csv`** — Excluded shelves
   - `shelf_raw`: Original shelf
   - `category`: Leakage category
   - `count`: Frequency
   - **Use**: Exclude these from semantic clustering

6. **`shelf_normalization_log.jsonl`** — Detailed audit log
   - JSON Lines format
   - Full provenance tracking
   - Decision rationale
   - Performance metrics

## Algorithm Details

### Canonicalization (2.1.A)

1. **Unicode Normalization**: NFKC form
2. **Separator Standardization**: Hyphens/underscores → spaces
3. **Edge Punctuation Removal**: Strip leading/trailing non-alphanumeric
4. **Whitespace Collapse**: Multiple spaces → single space
5. **Diacritic Removal**: NFD decomposition + combining character removal
6. **Case Folding**: Lowercase for comparison

### Segmentation (2.2)

**DFKI Conservative Approach**:

1. **CamelCase Detection**: `[A-Z][a-z]+[A-Z]` pattern
2. **Concatenation Detection**: Long lowercase runs (`^[a-z]{6,}$`)
3. **Dynamic Programming**: Word frequency-based segmentation
4. **Guard Conditions**:
   - **Guard 1**: At least one segment occurs as standalone shelf
   - **Guard 2**: All segments pass lexicon validation

### Alias Detection (2.1.B)

**Multi-metric Approach**:

1. **Jaro-Winkler Similarity**: String similarity (default: ≥0.94)
2. **Edit Distance**: Damerau-Levenshtein (default: ≤1)
3. **Character N-grams**: 3-gram Jaccard (default: ≥0.80)
4. **Blocking**: First token + trigram hash for efficiency
5. **Non-content Filtering**: Exclude leakage patterns

## Configuration

### Shelf Filters (`shelf_filters.yml`)

Categories of non-content shelves:

- **`ratings_valence`**: Star ratings, numeric scores
- **`process_status`**: Reading status (to-read, currently-reading)
- **`dates_campaigns`**: Year-based reading challenges
- **`format_edition`**: Book formats (hardcover, ebook)
- **`source_acquisition`**: Purchase sources (amazon, library)
- **`personal_org`**: Personal organization (favorites, wishlist)
- **`generic_noncontent`**: Empty strings, pure numbers

### Thresholds

- **`--jw-threshold`**: Jaro-Winkler similarity (default: 0.94)
- **`--j3-threshold`**: 3-gram Jaccard similarity (default: 0.80)
- **`--edit-max`**: Maximum edit distance (default: 1)
- **`--min-seg-evidence`**: Minimum frequency for domain lexicon (default: 5)
- **`--zipf-min`**: Minimum Zipf frequency for word validation (default: 3.0)

## Performance

### Scalability

- **Memory**: ~2GB for 50K books with 1M+ unique shelves
- **Time**: ~5-10 minutes for full dataset
- **Bottlenecks**: String similarity computation, parquet I/O

### Optimization

- **Top-K Limiting**: Use `--n-top-shelves` for large datasets
- **Blocking**: Efficient pairwise comparison via bucketing
- **Streaming**: PyArrow for memory-efficient parquet reading

## Testing

```bash
# Run test suite
python test_shelf_normalize.py

# Or via Makefile
make test-shelf
```

Test coverage:
- ✅ Canonicalization function
- ✅ Filter loading and matching
- ✅ End-to-end pipeline
- ✅ Output file validation
- ✅ Edge case handling

## Integration

### With Step 1 (Audit)

The pipeline consumes audit outputs:
- Schema validation results
- Data quality metrics
- Provenance information

### With Step 3 (Semantic Clustering)

The pipeline produces:
- Canonical shelf mappings
- Segment vocabulary
- Non-content exclusions
- Alias suggestions

## Troubleshooting

### Common Issues

1. **YAML Parsing Errors**: Check regex escaping in `shelf_filters.yml`
2. **Memory Issues**: Reduce `--n-top-shelves` or increase system memory
3. **Empty Alias Candidates**: Relax thresholds (`--jw-threshold`, `--j3-threshold`)
4. **Missing Dependencies**: Install optional packages for better performance

### Debug Mode

```bash
python shelf_normalize.py --verbose-logs [other options]
```

### Log Analysis

```bash
# Check processing stages
grep '"stage":' shelf_norm_outputs/shelf_normalization_log.jsonl | sort | uniq -c

# Find segmentation decisions
grep '"stage": "segment"' shelf_norm_outputs/shelf_normalization_log.jsonl | head -10

# Check alias candidates
grep '"stage": "alias"' shelf_norm_outputs/shelf_normalization_log.jsonl | head -10
```

## References

- **DFKI Guidelines**: Conservative segmentation approach
- **Clauset et al. (2009)**: Heavy-tail analysis methodology
- **Dean & Lawless (1989)**: Overdispersion testing
- **Cameron & Trivedi (1990)**: Auxiliary regression tests

## Next Steps

1. **Semantic Clustering**: Use canonical shelves and segments for embedding
2. **Alias Resolution**: Apply alias suggestions with book overlap validation
3. **Quality Assessment**: Evaluate normalization quality via human annotation
4. **Performance Tuning**: Optimize thresholds based on domain expertise
