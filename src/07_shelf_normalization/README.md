# Stage 07: Shelf Normalization

## Purpose

This stage normalizes user-generated shelf tags into canonical forms for analysis.

## Structure

```
07_shelf_normalization/
├── core/                          # Core normalization logic
│   ├── shelf_normalize.py        # Main normalization script
│   ├── simple_shelf_cleaner.py   # Simple cleaning utilities
│   ├── hybrid_classifier.py      # Hybrid classification approach
│   ├── simple_semantic_cluster.py # Semantic clustering
│   └── extract_shelves.py        # Shelf extraction utilities
├── bridge/                        # Integration with other pipeline steps
│   └── bridge_audit_normalize.py # Bridge normalization with parsed data
├── diagnostics/                   # Quality assurance and validation
│   ├── diagnostics_explore.py    # Diagnostic exploration
│   └── validate_bridge.py       # Bridge output validation
├── scripts/                       # Execution scripts
│   └── Makefile                  # Pipeline execution Makefile
├── docs/                          # Documentation
│   ├── LABELLING_GUIDE.md        # Labelling guide
│   ├── SEMANTIC_CLUSTERING_RESULTS.md
│   ├── SHELF_NORMALIZATION_IMPROVEMENTS.md
│   ├── SIMPLE_IMPROVEMENTS.md
│   └── TESTING_RESULTS.md
├── config/                        # Configuration files
├── tests/                         # Test files
├── outputs/                       # Stage outputs
├── __init__.py                    # Module exports
├── README.md                      # This file
├── README_SCIENTIFIC.md           # Scientific documentation
└── requirements.txt              # Python dependencies
```

## Input Files

- `data/processed/*.csv` - Datasets with shelf tags
- Audit outputs from Stage 02

## Output Files

- `data/processed/*_normalized.csv` - Datasets with normalized shelves
- `outputs/reports/*_shelf_normalization_*.json` - Normalization reports

## How to Run

### Complete Pipeline

**Option 1: Run from stage root**
```bash
cd src/07_shelf_normalization
make -f scripts/Makefile pipeline
```

**Option 2: Run from scripts directory**
```bash
cd src/07_shelf_normalization/scripts
make pipeline
```

### Individual Steps

```bash
# Step 1: Normalize shelves
make -f scripts/Makefile normalize

# Step 2: Bridge with parsed data
make -f scripts/Makefile bridge

# Step 3: Run diagnostics
make -f scripts/Makefile diagnostics

# Step 4: Validate outputs
make -f scripts/Makefile validate
```

## Dependencies

- pandas
- numpy
- Make (for Makefile)
- jaro-winkler (for similarity metrics)

## Example Usage

```bash
# Run complete normalization pipeline (from stage root)
make -f scripts/Makefile pipeline

# Or run from scripts directory
cd scripts
make pipeline

# Individual steps
make -f scripts/Makefile normalize
make -f scripts/Makefile bridge
make -f scripts/Makefile diagnostics
make -f scripts/Makefile validate
```

## Key Features

- Canonicalization of shelf strings
- CamelCase segmentation
- Alias detection
- Non-content filtering
- Quality assurance diagnostics
