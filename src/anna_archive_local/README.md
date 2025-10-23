# Anna's Archive Local Data Pipeline

An offline book search system using Anna's Archive data dumps, providing fast SQL-based queries without web scraping limitations.

## 🎯 Overview

This pipeline converts Anna's Archive ElasticSearch JSON dumps to Parquet format and uses DuckDB for efficient book searching. It's designed as an alternative to the existing web-scraping approach in `anna_archive_matcher`, offering:

- **No rate limiting**: Search locally without web restrictions
- **Faster queries**: SQL-based search on indexed data  
- **Offline capability**: Work without internet after setup
- **Bulk processing**: Handle thousands of books efficiently
- **Reliable**: No dependency on website availability

## 🚀 Quick Start

### 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Create data directories (already done)
mkdir -p ../../data/anna_archive/{elasticsearch,parquet}
```

### 2. Sample Data Workflow

```bash
# Extract sample from full dumps (if you have them)
python sample_data_extractor.py \
  --input-dir ../../data/anna_archive/elasticsearch/ \
  --output-file ../../data/anna_archive/elasticsearch/sample_10k.json.gz \
  --sample-size 10000

# Convert to Parquet
python json_to_parquet.py \
  --input-file ../../data/anna_archive/elasticsearch/sample_10k.json.gz \
  --output-dir ../../data/anna_archive/parquet/sample_10k/

# Search for books
python book_search_cli.py \
  --parquet-dir ../../data/anna_archive/parquet/sample_10k/ \
  --title "Fifty Shades" \
  --author "E.L. James"

# Run complete demo
python demo_query_50_books.py \
  --parquet-dir ../../data/anna_archive/parquet/sample_10k/ \
  --books-csv ../../data/processed/sample_50_books.csv \
  --output-dir ../../data/anna_archive/demo_results/
```

## 📁 Project Structure

```
src/anna_archive_local/
├── __init__.py                 # Module initialization
├── README.md                   # This file
├── DATA_ACQUISITION.md         # How to download Anna's Archive dumps
├── SAMPLING_GUIDE.md          # Working with data samples
├── requirements.txt            # Python dependencies
├── sample_data_extractor.py    # Extract samples from full dumps
├── json_to_parquet.py         # Convert JSON to Parquet
├── query_engine.py            # DuckDB search interface
├── book_search_cli.py         # Command-line search tool
├── demo_query_50_books.py     # Complete demo script
└── api_downloader.py          # Fast download API client

data/anna_archive/
├── elasticsearch/             # Raw JSON.gz files
├── elasticsearchAux/          # Auxiliary metadata
├── aac/                       # AAC data (optional)
├── mariadb/                   # SQL dumps (optional)
└── parquet/                   # Converted Parquet files
```

## 🔧 Components

### Data Processing

#### `sample_data_extractor.py`
Extracts samples from full Anna's Archive dumps for testing.

```bash
# Extract 10,000 records for development
python sample_data_extractor.py \
  --input-dir ../../data/anna_archive/elasticsearch/ \
  --output-file ../../data/anna_archive/elasticsearch/sample_10k.json.gz \
  --sample-size 10000
```

**Features**:
- Chunked processing for memory efficiency
- Progress tracking with statistics
- Multiple file support
- Sample analysis

#### `json_to_parquet.py`
Converts JSON.gz files to Parquet format for efficient querying.

```bash
# Convert sample to Parquet
python json_to_parquet.py \
  --input-file ../../data/anna_archive/elasticsearch/sample_10k.json.gz \
  --output-dir ../../data/anna_archive/parquet/sample_10k/
```

**Features**:
- Smart schema detection (samples 10% of data)
- Chunked processing (~10MB chunks)
- Field threshold filtering (include fields in >50% of records)
- Multiple Parquet part files for scalability

### Query Engine

#### `query_engine.py`
DuckDB-based search interface with fuzzy matching.

```python
from query_engine import BookSearchEngine

# Initialize search engine
engine = BookSearchEngine('../../data/anna_archive/parquet/sample_10k/')

# Search for books
results = engine.search_by_title_author("Fifty Shades", "E.L. James")
print(f"Found {len(results)} matches")

# Get dataset statistics
stats = engine.get_stats()
print(f"Total records: {stats['total_records']:,}")
```

**Features**:
- Automatic schema detection and field mapping
- Fuzzy title/author matching with SQL LIKE
- MD5 hash lookup
- Random book sampling
- Dataset statistics

#### `book_search_cli.py`
Command-line interface for manual book searches.

```bash
# Search by title and author
python book_search_cli.py \
  --parquet-dir ../../data/anna_archive/parquet/sample_10k/ \
  --title "Romeo and Juliet" \
  --author "Shakespeare"

# Get dataset statistics
python book_search_cli.py \
  --parquet-dir ../../data/anna_archive/parquet/sample_10k/ \
  --stats

# Show random books
python book_search_cli.py \
  --parquet-dir ../../data/anna_archive/parquet/sample_10k/ \
  --random 5
```

### Download Integration

#### `api_downloader.py`
Fast download client using Anna's Archive API.

```python
from api_downloader import AnnaArchiveDownloader

# Initialize downloader
downloader = AnnaArchiveDownloader(api_key="your_api_key")

# Download book by MD5
result = downloader.download_book("md5_hash", "output_directory")
if result:
    print(f"Downloaded: {result}")
```

**Features**:
- Multiple authentication methods
- Retry logic with exponential backoff
- Progress tracking for large files
- Batch download support
- Error handling and cleanup

### Demo and Testing

#### `demo_query_50_books.py`
Complete end-to-end demo using the 50 sample books.

```bash
# Run complete demo
python demo_query_50_books.py \
  --parquet-dir ../../data/anna_archive/parquet/sample_10k/ \
  --books-csv ../../data/processed/sample_50_books.csv \
  --output-dir ../../data/anna_archive/demo_results/ \
  --api-key "your_api_key"
```

**Outputs**:
- `search_results.csv`: Detailed search results
- `search_summary.json`: Statistics and analysis
- `download_ready.csv`: Books with MD5 hashes ready for download
- `search_report.md`: Human-readable report
- `test_downloads/`: Sample downloaded books (if API key provided)

## 📊 Performance Metrics

### Expected Results

| Dataset Size | Records | Disk Space | RAM Required | Query Time | Match Rate |
|--------------|---------|------------|--------------|------------|------------|
| 1K Sample | 1,000 | ~50MB | 2GB | <1ms | 0-20% |
| 10K Sample | 10,000 | ~500MB | 4GB | <5ms | 10-30% |
| 50K Sample | 50,000 | ~2GB | 8GB | <20ms | 20-40% |
| Full Dataset | ~10M | ~500GB | 30GB | <100ms | 30-50% |

### Comparison: Local vs Web Scraping

| Aspect | Local Data Pipeline | Web Scraping (`anna_archive_matcher`) |
|--------|-------------------|--------------------------------------|
| **Speed** | Very fast (SQL queries) | Slow (HTTP requests + parsing) |
| **Rate Limits** | None | Yes (delays required) |
| **Reliability** | High (offline) | Medium (network dependent) |
| **Storage** | High (500GB+ full) | Low (minimal) |
| **Setup** | Complex (data acquisition) | Simple (just run) |
| **Scalability** | Excellent (bulk processing) | Limited (sequential) |
| **Maintenance** | Low (stable data) | High (site changes) |

## 🔄 Workflow Examples

### Development Workflow

```bash
# 1. Start with small sample
python sample_data_extractor.py \
  --input-dir ../../data/anna_archive/elasticsearch/ \
  --output-file ../../data/anna_archive/elasticsearch/sample_1k.json.gz \
  --sample-size 1000

# 2. Convert to Parquet
python json_to_parquet.py \
  --input-file ../../data/anna_archive/elasticsearch/sample_1k.json.gz \
  --output-dir ../../data/anna_archive/parquet/sample_1k/

# 3. Test queries
python book_search_cli.py \
  --parquet-dir ../../data/anna_archive/parquet/sample_1k/ \
  --stats

# 4. Run demo
python demo_query_50_books.py \
  --parquet-dir ../../data/anna_archive/parquet/sample_1k/ \
  --books-csv ../../data/processed/sample_50_books.csv \
  --output-dir ../../data/anna_archive/demo_results/
```

### Production Workflow

```bash
# 1. Download full dataset (see DATA_ACQUISITION.md)
# 2. Convert full dataset
python json_to_parquet.py \
  --input-dir ../../data/anna_archive/elasticsearch/ \
  --output-dir ../../data/anna_archive/parquet/full/

# 3. Search at scale
python book_search_cli.py \
  --parquet-dir ../../data/anna_archive/parquet/full/ \
  --title "your search" \
  --limit 100

# 4. Download found books
python api_downloader.py \
  --api-key "your_api_key" \
  --csv-file ../../data/anna_archive/demo_results/download_ready.csv \
  --output-dir ../../organized_outputs/epub_downloads/
```

## 🛠️ Configuration

### Memory Optimization

For systems with limited RAM:

```bash
# Reduce chunk sizes
python json_to_parquet.py \
  --chunk-size 5000 \
  --sample-size 500

# Use smaller samples
python sample_data_extractor.py \
  --sample-size 1000 \
  --max-files 5
```

### Performance Tuning

```bash
# Increase chunk size for faster processing
python json_to_parquet.py \
  --chunk-size 20000

# Adjust field threshold for more/fewer fields
python json_to_parquet.py \
  --field-threshold 0.3  # Include fields in >30% of records
```

## 🔍 Schema Information

### Common Fields

The pipeline automatically detects and maps these fields:

- **title**: Book title (from `_source.file_unified_data.title.best`)
- **author**: Author name (from `_source.file_unified_data.author.best`)
- **publisher**: Publisher name (from `_source.file_unified_data.publisher.best`)
- **year**: Publication year (from `_source.file_unified_data.year.best`)
- **md5**: File MD5 hash (from `_source.file_unified_data.identifiers_unified.md5`)
- **extension**: File format (from `_source.file_unified_data.extension.best`)
- **language**: Book language (from `_source.file_unified_data.language.best`)
- **file_size**: File size in bytes (from `_source.file_unified_data.size`)

### Field Mapping

The system uses regex patterns to map Anna's Archive's nested JSON structure to standard field names:

```python
patterns = {
    'title': [r'.*title.*best.*', r'.*_source.*file_unified_data.*title.*best.*'],
    'author': [r'.*author.*best.*', r'.*_source.*file_unified_data.*author.*best.*'],
    'md5': [r'.*md5.*', r'.*identifiers.*md5.*'],
    # ... more patterns
}
```

## 🚨 Troubleshooting

### Common Issues

1. **"No Parquet files found"**
   - Ensure you've run `json_to_parquet.py` first
   - Check that output directory contains `.parquet` files

2. **"Schema fields not detected"**
   - Try with a larger sample size
   - Check that JSON files are valid and contain expected structure

3. **"Memory errors during conversion"**
   - Reduce chunk size: `--chunk-size 5000`
   - Use smaller samples: `--sample-size 1000`
   - Increase system RAM or use swap

4. **"No books found in search"**
   - Try fuzzy matching (default)
   - Use partial titles/authors
   - Check if your sample contains the books you're searching for

5. **"API download failures"**
   - Verify API key is correct
   - Check network connectivity
   - Try different authentication methods

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
python book_search_cli.py --verbose --parquet-dir ... --title "test"
python demo_query_50_books.py --verbose --parquet-dir ... --books-csv ...
```

## 📚 Documentation

- **[DATA_ACQUISITION.md](DATA_ACQUISITION.md)**: How to download Anna's Archive data dumps
- **[SAMPLING_GUIDE.md](SAMPLING_GUIDE.md)**: Working with data samples for testing
- **[Existing anna_archive_matcher](../anna_archive_matcher/README.md)**: Web scraping approach

## 🤝 Integration

### With Existing Project

This local pipeline complements the existing `anna_archive_matcher` module:

- **Use local pipeline for**: Bulk processing, offline work, research analysis
- **Use web scraping for**: Real-time searches, small batches, when local data isn't available

### API Integration

The pipeline integrates with Anna's Archive's fast download API:

```python
# Search locally
from query_engine import BookSearchEngine
engine = BookSearchEngine('parquet_dir')
results = engine.search_by_title_author("title", "author")

# Download using API
from api_downloader import AnnaArchiveDownloader
downloader = AnnaArchiveDownloader(api_key="your_key")
for result in results:
    if result.get('md5'):
        downloader.download_book(result['md5'], 'output_dir')
```

## 🎯 Next Steps

1. **Test with samples**: Start with 1K-10K record samples
2. **Validate results**: Compare with existing web scraping results
3. **Scale up**: Process full dataset when ready
4. **Integrate**: Use in your research workflow
5. **Optimize**: Tune parameters for your specific needs

## 📄 License

This module is part of the Romance Novel NLP Research project. See the main project LICENSE for details.

## 🙏 Acknowledgments

- **Anna's Archive**: For providing public data dumps
- **RArtutos**: For the Data Science Starter Kit that inspired this approach
- **DuckDB**: For the excellent in-process SQL engine
- **PyArrow**: For efficient Parquet processing
