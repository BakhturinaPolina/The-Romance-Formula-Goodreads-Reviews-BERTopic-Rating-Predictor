# Anna's Archive Data Sampling Guide

## Overview

This guide explains how to work with samples of Anna's Archive data for testing and development. Working with full datasets (500GB+) can be resource-intensive, so sampling allows you to test the pipeline with smaller datasets first.

## Why Use Sampling?

- **Resource Efficiency**: Test with 1-5GB instead of 500GB+
- **Faster Development**: Quick iteration cycles
- **Lower Requirements**: Works on machines with 8GB RAM
- **Validation**: Verify pipeline works before full processing
- **Debugging**: Easier to debug issues with smaller datasets

## Sampling Strategies

### Strategy 1: First N Records (Recommended)
Extract the first N records from each JSON.gz file.

**Pros**: Simple, maintains file structure
**Cons**: May not be representative of full dataset
**Use Case**: Initial testing and development

### Strategy 2: Random Sampling
Randomly select records across all files.

**Pros**: More representative sample
**Cons**: More complex implementation
**Use Case**: Statistical analysis and validation

### Strategy 3: Genre-Based Sampling
Sample records based on specific criteria (e.g., romance books).

**Pros**: Targeted for specific research needs
**Cons**: Requires pre-processing to identify criteria
**Use Case**: Domain-specific research

## Implementation

### Sample Size Recommendations

| Purpose | Records | Disk Space | RAM Required | Processing Time |
|---------|---------|------------|--------------|-----------------|
| Quick Test | 1,000 | ~50MB | 2GB | 2-5 minutes |
| Development | 10,000 | ~500MB | 4GB | 10-20 minutes |
| Validation | 50,000 | ~2GB | 8GB | 30-60 minutes |
| Pre-production | 100,000 | ~5GB | 16GB | 1-2 hours |

### Using the Sample Extractor

The `sample_data_extractor.py` script provides easy sampling:

```bash
# Extract 10,000 records for development
python sample_data_extractor.py \
  --input-dir ../../data/anna_archive/elasticsearch/ \
  --output-file ../../data/anna_archive/elasticsearch/sample_10k.json.gz \
  --sample-size 10000

# Extract 1,000 records for quick testing
python sample_data_extractor.py \
  --input-dir ../../data/anna_archive/elasticsearch/ \
  --output-file ../../data/anna_archive/elasticsearch/sample_1k.json.gz \
  --sample-size 1000
```

## Sample Data Structure

After sampling, you'll have:

```
data/anna_archive/
├── elasticsearch/
│   ├── sample_1k.json.gz      # 1,000 records for quick testing
│   ├── sample_10k.json.gz     # 10,000 records for development
│   └── sample_50k.json.gz     # 50,000 records for validation
└── parquet/
    ├── sample_1k/             # Converted Parquet files
    ├── sample_10k/
    └── sample_50k/
```

## Processing Samples

### Convert to Parquet

```bash
# Process 1K sample
python json_to_parquet.py \
  --input-file ../../data/anna_archive/elasticsearch/sample_1k.json.gz \
  --output-dir ../../data/anna_archive/parquet/sample_1k/

# Process 10K sample
python json_to_parquet.py \
  --input-file ../../data/anna_archive/elasticsearch/sample_10k.json.gz \
  --output-dir ../../data/anna_archive/parquet/sample_10k/
```

### Query Samples

```python
from query_engine import BookSearchEngine

# Initialize with sample data
engine = BookSearchEngine('../../data/anna_archive/parquet/sample_10k/')

# Search for a book
results = engine.search_by_title_author("Fifty Shades", "E.L. James")
print(f"Found {len(results)} matches")
```

## Testing with Romance Books

### Sample Romance Books for Testing

Use the provided `sample_50_books.csv` to test your pipeline:

```python
import pandas as pd
from query_engine import BookSearchEngine

# Load test books
books = pd.read_csv('../../data/processed/sample_50_books.csv')

# Initialize search engine
engine = BookSearchEngine('../../data/anna_archive/parquet/sample_10k/')

# Test search for each book
results = []
for _, book in books.iterrows():
    matches = engine.search_by_title_author(book['title'], book['author_name'])
    results.append({
        'title': book['title'],
        'author': book['author_name'],
        'found': len(matches) > 0,
        'matches': len(matches)
    })

# Analyze results
df_results = pd.DataFrame(results)
print(f"Success rate: {df_results['found'].mean():.1%}")
```

## Performance Expectations

### Sample vs Full Dataset Performance

| Metric | 1K Sample | 10K Sample | 50K Sample | Full Dataset |
|--------|-----------|------------|------------|--------------|
| Query Time | <1ms | <5ms | <20ms | <100ms |
| Memory Usage | <100MB | <500MB | <2GB | <10GB |
| Disk I/O | Minimal | Low | Moderate | High |
| Match Rate | Variable | More stable | Representative | Full coverage |

### Expected Match Rates

For romance books in samples:
- **1K sample**: 0-20% (may miss popular books)
- **10K sample**: 10-30% (better coverage)
- **50K sample**: 20-40% (good representation)
- **Full dataset**: 30-50% (comprehensive)

## Scaling Up

### From Sample to Full Dataset

1. **Validate Pipeline**: Ensure everything works with samples
2. **Resource Planning**: Verify sufficient disk space and RAM
3. **Batch Processing**: Process full dataset in chunks
4. **Monitor Performance**: Watch for memory/disk issues
5. **Backup Strategy**: Keep samples for quick testing

### Migration Steps

```bash
# 1. Test with small sample
python demo_query_50_books.py --sample-size 1000

# 2. Validate with medium sample  
python demo_query_50_books.py --sample-size 10000

# 3. Process full dataset
python json_to_parquet.py --input-dir ../../data/anna_archive/elasticsearch/

# 4. Run full test
python demo_query_50_books.py --use-full-dataset
```

## Troubleshooting Samples

### Common Issues

1. **Empty Results**
   - Sample too small for your test books
   - Try larger sample size
   - Check if test books exist in Anna's Archive

2. **Memory Errors**
   - Reduce sample size
   - Increase system RAM
   - Use chunked processing

3. **Slow Processing**
   - Use smaller samples for development
   - Optimize chunk sizes
   - Use faster storage (SSD)

### Debugging Tips

```python
# Check sample contents
import json
import gzip

with gzip.open('sample_1k.json.gz', 'rt') as f:
    for i, line in enumerate(f):
        if i >= 5:  # Show first 5 records
            break
        record = json.loads(line)
        print(f"Record {i}: {record.get('_source', {}).get('file_unified_data', {}).get('title', {}).get('best', 'No title')}")
```

## Best Practices

1. **Start Small**: Begin with 1K samples for initial testing
2. **Iterate Quickly**: Use 10K samples for development
3. **Validate Thoroughly**: Test with 50K samples before full processing
4. **Keep Samples**: Maintain sample datasets for quick testing
5. **Document Results**: Track performance metrics across sample sizes

## Next Steps

After successful sampling:

1. **Run Demo**: Execute `demo_query_50_books.py` with samples
2. **Analyze Results**: Review match rates and performance
3. **Optimize Pipeline**: Adjust parameters based on sample results
4. **Scale Up**: Process full dataset when ready
5. **Production Use**: Deploy for your research needs
