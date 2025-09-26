# Bridge Script: Step 1 → Step 2 Integration

**File:** `02_bridge_audit_normalize.py`

This script bridges the outputs from Step 1 (audit + parsed shelves) and Step 2 (shelf normalization) into an integrated dataset ready for clustering and modeling.

## Overview

The bridge script takes the parsed books data from Step 1 and applies the normalization artifacts from Step 2 to create a comprehensive dataset with:

- **Canonical shelf mappings** (raw → normalized)
- **Content filtering** (removes non-content shelves)
- **Segment tokenization** (splits compound shelves)
- **Multiple output formats** (wide and long tables)

## Inputs

### Required Inputs

1. **Parsed Books** (`--parsed`): Parquet file from `01_parse_lists.py`
   - Contains `shelves` column with list of shelf strings per book
   - Example: `parse_outputs_full/parsed_books_20250926_002946.parquet`

2. **Canonical Mapping** (`--canon`): CSV from `shelf_normalize.py`
   - Maps raw shelf strings to canonical forms
   - Columns: `shelf_raw`, `shelf_canon`, `reason`, `count`
   - Example: `normalize_outputs/shelf_canonical.csv`

3. **Segments Data** (`--segments`): CSV from `shelf_normalize.py`
   - Segmentation results for canonical shelves
   - Columns: `shelf_canon`, `segments`, `accepted`, `evidence_count`
   - Example: `normalize_outputs/shelf_segments.csv`

4. **Non-Content Filter** (`--noncontent`): CSV from `shelf_normalize.py`
   - Identifies non-content shelves to exclude
   - Columns: `shelf_raw`, `category`, `count`
   - Example: `normalize_outputs/noncontent_shelves.csv`

### Optional Inputs

5. **Alias Candidates** (`--alias`): CSV from `shelf_normalize.py`
   - Potential shelf aliases for quality control
   - Columns: `shelf_a`, `shelf_b`, `jw`, `edit`, `jaccard3`, `decision_hint`
   - Example: `normalize_outputs/shelf_alias_candidates.csv`

## Outputs

All outputs are written to the specified `--out` directory:

### Core Outputs

1. **`books_with_shelf_norm.parquet`** — Enhanced books dataset
   - Original book data + normalized shelf columns
   - `shelves_canon`: Canonical shelf lists
   - `shelves_canon_content`: Content-only shelf lists (non-content filtered)
   - `shelves_noncontent_flags`: Boolean flags for non-content shelves
   - `n_shelves_*`: Count columns for each shelf type

2. **`shelves_raw_long.parquet`** — Raw shelves in long format
   - One row per book-shelf combination
   - Columns: `work_id`, `row_index`, `shelf_raw`

3. **`shelves_canon_long.parquet`** — Canonical shelves in long format
   - One row per book-canonical shelf combination
   - Columns: `work_id`, `row_index`, `shelf_canon`

4. **`segments_long.parquet`** — Segments in long format
   - One row per book-segment combination
   - Columns: `work_id`, `row_index`, `shelf_canon`, `segment`, `seg_accepted`

5. **`bridge_summary.json`** — Processing summary
   - Total counts, unique counts, averages
   - Processing statistics and metadata

6. **`shelf_norm_bridge_log.jsonl`** — Provenance log
   - JSON Lines format with full processing trace
   - Input/output paths, timestamps, environment info

## Usage

### Basic Usage

```bash
python 02_bridge_audit_normalize.py \
  --parsed parse_outputs_full/parsed_books_20250926_002946.parquet \
  --canon normalize_outputs/shelf_canonical.csv \
  --segments normalize_outputs/shelf_segments.csv \
  --noncontent normalize_outputs/noncontent_shelves.csv \
  --out bridge_outputs/
```

### With Alias Candidates

```bash
python 02_bridge_audit_normalize.py \
  --parsed parse_outputs_full/parsed_books_20250926_002946.parquet \
  --canon normalize_outputs/shelf_canonical.csv \
  --segments normalize_outputs/shelf_segments.csv \
  --noncontent normalize_outputs/noncontent_shelves.csv \
  --alias normalize_outputs/shelf_alias_candidates.csv \
  --out bridge_outputs/
```

### Custom Column Names

```bash
python 02_bridge_audit_normalize.py \
  --parsed parse_outputs_full/parsed_books_20250926_002946.parquet \
  --canon normalize_outputs/shelf_canonical.csv \
  --segments normalize_outputs/shelf_segments.csv \
  --noncontent normalize_outputs/noncontent_shelves.csv \
  --out bridge_outputs/ \
  --id-col book_id \
  --shelves-col shelf_tags
```

## Data Flow

```
Step 1 Outputs                    Step 2 Outputs
├── parsed_books.parquet    +     ├── shelf_canonical.csv
└── (shelves lists)               ├── shelf_segments.csv
                                  ├── noncontent_shelves.csv
                                  └── shelf_alias_candidates.csv
                                          ↓
                                  Bridge Script
                                          ↓
                              Integrated Dataset
├── books_with_shelf_norm.parquet
├── shelves_raw_long.parquet
├── shelves_canon_long.parquet
├── segments_long.parquet
├── bridge_summary.json
└── shelf_norm_bridge_log.jsonl
```

## Key Features

### **Robust Data Handling**
- Handles both pyarrow and fastparquet engines
- Graceful fallbacks for missing dependencies
- Case-insensitive key matching
- Safe list parsing and validation

### **Content Filtering**
- Applies non-content shelf filters
- Preserves original data alongside filtered versions
- Boolean flags for transparency

### **Multiple Output Formats**
- **Wide format**: All data in one table with list columns
- **Long formats**: One row per shelf/segment for analysis
- **Summary**: Aggregated statistics
- **Provenance**: Full processing trace

### **Production Ready**
- Comprehensive error handling
- Detailed logging and provenance
- Configurable column names
- Idempotent operations

## Testing

The script includes a comprehensive test suite:

```bash
# Run all tests
pytest test_02_bridge_audit_normalize.py -v

# Run specific test categories
pytest test_02_bridge_audit_normalize.py::TestBridgeAuditNormalize::test_end_to_end_pipeline -v
pytest test_02_bridge_audit_normalize.py::TestBridgeAuditNormalize::test_idempotence -v
```

Test coverage includes:
- ✅ Schema validation and row count consistency
- ✅ Idempotence (reproducible results)
- ✅ Edge case handling (empty files, malformed data)
- ✅ Data integrity (canonicalization, filtering accuracy)
- ✅ Output format validation

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

## Troubleshooting

### Common Issues

1. **Missing Input Files**: Ensure all required Step 1 and Step 2 outputs exist
2. **Column Mismatches**: Check that input files have expected column names
3. **Memory Issues**: Use smaller datasets or increase system memory
4. **Parquet Engine Issues**: Script automatically falls back between pyarrow/fastparquet

### Debug Mode

```bash
python 02_bridge_audit_normalize.py --verbose-logs [other options]
```

### Log Analysis

```bash
# Check processing summary
cat bridge_outputs/bridge_summary.json | jq .

# Check provenance log
tail -n 1 bridge_outputs/shelf_norm_bridge_log.jsonl | jq .
```

## Dependencies

- `pandas>=2.0` — Data manipulation
- `pyarrow>=10.0` or `fastparquet` — Parquet file handling
- `numpy>=1.20` — Numerical operations
- `pathlib` — Path handling (built-in)

## Next Steps

1. **Semantic Clustering**: Use canonical shelves and segments for embedding
2. **Alias Resolution**: Apply alias suggestions with book overlap validation
3. **Quality Assessment**: Evaluate normalization quality via human annotation
4. **Performance Tuning**: Optimize based on downstream analysis needs
