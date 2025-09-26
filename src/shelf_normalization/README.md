# Shelf Normalization Pipeline

**Step 2 of the Romance Novel EDA Analysis Pipeline**

This module implements the shelf normalization pipeline that transforms messy shelf tags into normalized, canonical forms suitable for downstream analysis. It bridges the audit outputs (Step 1) to semantic shelf clustering (Step 3).

## Overview

The shelf normalization pipeline performs three main operations:

1. **Canonicalization**: Deterministic normalization of shelf strings
2. **Segmentation**: Conservative CamelCase and concatenation splitting  
3. **Alias Detection**: Identification of potential shelf aliases

## Project Structure

```
src/shelf_normalization/
├── core/                          # Core normalization logic
│   └── shelf_normalize.py        # Main pipeline script
├── bridge/                        # Integration with other pipeline steps
│   └── bridge_audit_normalize.py # Bridge Step 1 → Step 2
├── diagnostics/                   # Quality assurance and validation
│   ├── diagnostics_explore.py    # Deep-dive diagnostics
│   └── validate_bridge.py        # Bridge output validation
├── config/                        # Configuration files (future)
├── tests/                         # Test suite (future)
├── requirements.txt               # Dependencies
├── Makefile                      # Easy pipeline execution
└── README.md                     # This file
```

## Dependencies

```bash
pip install -r requirements.txt
```

**Core dependencies:**
- `pandas>=2.0` — Data manipulation
- `pyarrow>=10.0` — Parquet file handling
- `numpy>=1.23` — Numerical operations

**Optional but recommended:**
- `rapidfuzz>=2.0` — String similarity metrics (faster)
- `wordfreq>=3.0` — Word frequency data (better segmentation)

## Quick Start

### 1. Setup

```bash
# Install dependencies
make install

# Create output directories
make setup
```

### 2. Run Complete Pipeline

```bash
# Run all steps in sequence
make pipeline
```

### 3. Individual Steps

```bash
# Step 1: Normalize shelves
make normalize

# Step 2: Bridge with parsed data
make bridge

# Step 3: Run diagnostics
make diagnostics

# Step 4: Validate outputs
make validate
```

## Detailed Usage

### Step 1: Shelf Normalization

**Script:** `core/shelf_normalize.py`

Transforms raw shelf strings into canonical forms and identifies potential aliases.

```bash
python core/shelf_normalize.py \
    --from-audit ../../data/processed/audit_outputs \
    --parsed-parquet ../../data/processed/parsed_books.parquet \
    --outdir outputs/normalize \
    --n-top-shelves 100000 \
    --jw-threshold 0.94 \
    --j3-threshold 0.80 \
    --edit-max 1 \
    --min-seg-evidence 5 \
    --zipf-min 3.0
```

**Inputs:**
- `--from-audit`: Directory containing `audit_results.json` and `overdispersion_tests.json`
- `--parsed-parquet`: Parsed books parquet from Step 1 with `shelves` column

**Outputs:**
- `shelf_canonical.csv` — Raw → canonical mapping
- `shelf_segments.csv` — Segmentation results
- `shelf_alias_candidates.csv` — Potential aliases
- `noncontent_shelves.csv` — Excluded non-content shelves
- `segments_vocab.txt` — Unique segment vocabulary
- `shelf_normalization_log.jsonl` — Detailed audit log

### Step 2: Bridge Integration

**Script:** `bridge/bridge_audit_normalize.py`

Integrates normalization results with parsed book data.

```bash
python bridge/bridge_audit_normalize.py \
    --parsed ../../data/processed/parsed_books.parquet \
    --canon outputs/normalize/shelf_canonical.csv \
    --segments outputs/normalize/shelf_segments.csv \
    --noncontent outputs/normalize/noncontent_shelves.csv \
    --alias outputs/normalize/shelf_alias_candidates.csv \
    --out outputs/bridge
```

**Outputs:**
- `books_with_shelf_norm.parquet` — Enhanced books dataset
- `shelves_raw_long.parquet` — Raw shelves in long format
- `shelves_canon_long.parquet` — Canonical shelves in long format
- `segments_long.parquet` — Segments in long format
- `bridge_summary.json` — Processing summary
- `shelf_norm_bridge_log.jsonl` — Provenance log

### Step 3: Diagnostics

**Script:** `diagnostics/diagnostics_explore.py`

Deep-dive analysis of normalization quality.

```bash
python diagnostics/diagnostics_explore.py \
    --in-dir outputs/normalize \
    --out-dir outputs/diagnostics \
    --alias-dryrun-size 10000 \
    --alias-topk 200 \
    --verbose-logs
```

**Outputs:**
- `diagnostics_report.txt` — Human-readable report
- `diagnostics_summary.json` — Machine-readable summary
- `alias_dryrun_samples.csv` — Sample shelves for alias analysis
- `alias_dryrun_top_pairs.csv` — Top alias candidate pairs

### Step 4: Validation

**Script:** `diagnostics/validate_bridge.py`

Comprehensive validation of bridge outputs.

```bash
python diagnostics/validate_bridge.py \
    --books outputs/bridge/books_with_shelf_norm.parquet \
    --raw-long outputs/bridge/shelves_raw_long.parquet \
    --canon-long outputs/bridge/shelves_canon_long.parquet \
    --segments-long outputs/bridge/segments_long.parquet \
    --canonical-csv outputs/normalize/shelf_canonical.csv \
    --segments-csv outputs/normalize/shelf_segments.csv \
    --noncontent-csv outputs/normalize/noncontent_shelves.csv \
    --alias-csv outputs/normalize/shelf_alias_candidates.csv \
    --out outputs/diagnostics/bridge_qa \
    --print-examples \
    --verbose-logs
```

**Outputs:**
- `diagnostics_summary.json` — Validation results
- `diagnostics_report.txt` — Human-readable report
- `suspect_examples_*.csv` — Sample problematic rows

## Algorithm Details

### Canonicalization

1. **Unicode Normalization**: NFKC form
2. **Separator Standardization**: Hyphens/underscores → spaces
3. **Edge Punctuation Removal**: Strip leading/trailing non-alphanumeric
4. **Whitespace Collapse**: Multiple spaces → single space
5. **Diacritic Removal**: NFD decomposition + combining character removal
6. **Case Folding**: Lowercase for comparison

### Segmentation

**Conservative Approach:**
1. **CamelCase Detection**: `[A-Z][a-z]+[A-Z]` pattern
2. **Concatenation Detection**: Long lowercase runs (`^[a-z]{6,}$`)
3. **Guard Conditions**:
   - **Guard 1**: At least one segment occurs as standalone shelf
   - **Guard 2**: All segments pass lexicon validation

### Alias Detection

**Multi-metric Approach:**
1. **Jaro-Winkler Similarity**: String similarity (default: ≥0.94)
2. **Edit Distance**: Damerau-Levenshtein (default: ≤1)
3. **Character N-grams**: 3-gram Jaccard (default: ≥0.80)
4. **Blocking**: First character for efficiency
5. **Non-content Filtering**: Exclude leakage patterns

### Non-Content Filtering

Categories of excluded shelves:
- **`ratings_valence`**: Star ratings, numeric scores
- **`process_status`**: Reading status (to-read, currently-reading)
- **`dates_campaigns`**: Year-based reading challenges
- **`format_edition`**: Book formats (hardcover, ebook)
- **`source_acquisition`**: Purchase sources (amazon, library)
- **`personal_org`**: Personal organization (favorites, wishlist)
- **`generic_noncontent`**: Empty strings, pure numbers

## Configuration

### Makefile Variables

Edit the Makefile to adjust default paths:

```makefile
AUDIT_DIR = ../../data/processed/audit_outputs
PARSED_PARQUET = ../../data/processed/parsed_books.parquet
NORMALIZE_OUTDIR = outputs/normalize
BRIDGE_OUTDIR = outputs/bridge
DIAGNOSTICS_OUTDIR = outputs/diagnostics
```

### Parameter Tuning

**Normalization Parameters:**
- `--n-top-shelves`: Limit processing to top-K shelves (default: 100k)
- `--jw-threshold`: Jaro-Winkler similarity threshold (default: 0.94)
- `--j3-threshold`: 3-gram Jaccard threshold (default: 0.80)
- `--edit-max`: Maximum edit distance (default: 1)
- `--min-seg-evidence`: Minimum evidence for segmentation (default: 5)
- `--zipf-min`: Minimum Zipf frequency for lexicon validation (default: 3.0)

## Quality Assurance

### Built-in Validation

The pipeline includes comprehensive quality checks:

1. **Schema Validation**: Required columns present
2. **Data Integrity**: List length consistency
3. **Coverage Analysis**: Canonical mapping completeness
4. **Flag Alignment**: Non-content filtering accuracy
5. **Unicode Safety**: Control character detection
6. **Duplicate Detection**: Upstream duplication identification

### Diagnostic Reports

**Human-readable reports** include:
- Canonicalization compression ratios
- Segmentation acceptance rates
- Alias candidate quality metrics
- Non-content filtering effectiveness
- Vocabulary size and sample tokens

**Machine-readable summaries** for:
- Automated quality monitoring
- Regression testing
- Performance benchmarking

## Troubleshooting

### Common Issues

1. **Missing Input Files**: Ensure all required Step 1 outputs exist
2. **Memory Issues**: Reduce `--n-top-shelves` parameter
3. **Parquet Engine Issues**: Script automatically falls back between pyarrow/fastparquet
4. **Unicode Errors**: Check for malformed text in input data

### Debug Mode

```bash
# Enable verbose logging
python core/shelf_normalize.py --verbose-logs [other options]

# Check processing logs
tail -f outputs/normalize/shelf_normalization_log.jsonl

# Validate outputs
make validate
```

### Performance Optimization

1. **Install optional dependencies** for better performance:
   ```bash
   pip install rapidfuzz wordfreq
   ```

2. **Adjust batch sizes** for large datasets:
   ```bash
   python core/shelf_normalize.py --n-top-shelves 50000 [other options]
   ```

3. **Use SSD storage** for temporary files

## Integration

### With Step 1 (Audit & Parsing)
- Consumes parsed books with list columns intact
- Preserves all original book metadata
- Adds shelf normalization columns

### With Step 3 (Semantic Clustering)
- Provides canonical shelf mappings
- Segment vocabulary for embedding
- Content-only shelves for clustering
- Long formats for analysis

## Testing

### Manual Testing

```bash
# Test individual components
python core/shelf_normalize.py --help
python bridge/bridge_audit_normalize.py --help
python diagnostics/diagnostics_explore.py --help
python diagnostics/validate_bridge.py --help
```

### Quality Validation

```bash
# Run complete pipeline with validation
make pipeline

# Check outputs
ls -la outputs/normalize/
ls -la outputs/bridge/
ls -la outputs/diagnostics/
```

## Output Files Reference

### Core Outputs

| File | Description | Columns |
|------|-------------|---------|
| `shelf_canonical.csv` | Raw → canonical mapping | `shelf_raw`, `shelf_canon`, `reason`, `count` |
| `shelf_segments.csv` | Segmentation results | `shelf_canon`, `segments`, `accepted`, `guard1_standalone`, `guard2_lexicon`, `evidence_count` |
| `shelf_alias_candidates.csv` | Potential aliases | `shelf_a`, `shelf_b`, `jw`, `edit`, `jaccard3`, `decision_hint` |
| `noncontent_shelves.csv` | Excluded shelves | `shelf_raw`, `shelf_canon`, `category`, `count` |
| `segments_vocab.txt` | Unique segment vocabulary | One token per line |

### Bridge Outputs

| File | Description | Columns |
|------|-------------|---------|
| `books_with_shelf_norm.parquet` | Enhanced books dataset | Original + `shelves_canon`, `shelves_canon_content`, `shelves_noncontent_flags`, `n_shelves_*` |
| `shelves_raw_long.parquet` | Raw shelves (long) | `work_id`, `row_index`, `shelf_raw` |
| `shelves_canon_long.parquet` | Canonical shelves (long) | `work_id`, `row_index`, `shelf_canon` |
| `segments_long.parquet` | Segments (long) | `work_id`, `row_index`, `shelf_canon`, `segment`, `seg_accepted` |

### Diagnostic Outputs

| File | Description | Content |
|------|-------------|---------|
| `diagnostics_report.txt` | Human-readable report | Canonicalization, segmentation, alias analysis |
| `diagnostics_summary.json` | Machine-readable summary | Quantitative metrics and statistics |
| `alias_dryrun_samples.csv` | Sample shelves | Reproducible alias analysis samples |
| `alias_dryrun_top_pairs.csv` | Top alias pairs | Highest-scoring alias candidates |

## Contributing

### Code Organization

- **Core logic** in `core/` directory
- **Integration** in `bridge/` directory  
- **Quality assurance** in `diagnostics/` directory
- **Configuration** in `config/` directory (future)
- **Tests** in `tests/` directory (future)

### Development Guidelines

1. **Follow coding agent patterns** from `.cursor/rules/coding-agent-pattern.mdc`
2. **Maintain non-destructive edits** per `.cursor/rules/guardrails-non-destructive-edits.mdc`
3. **Add comprehensive logging** for debugging
4. **Include provenance tracking** in all outputs
5. **Validate inputs and outputs** thoroughly

### Adding New Features

1. **Create feature branch**: `git checkout -b feat/new-feature`
2. **Implement with tests**: Add validation and diagnostics
3. **Update documentation**: Include in README and help text
4. **Test integration**: Ensure compatibility with existing pipeline
5. **Submit pull request**: Include comprehensive description

## License

This project is part of the Romance Novel NLP Research project. See the main project LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review diagnostic outputs for error details
3. Enable verbose logging for debugging
4. Check the main project documentation

---

**Last Updated**: 2025-01-09  
**Version**: 1.0.0  
**Maintainer**: Romance Novel NLP Research Team
