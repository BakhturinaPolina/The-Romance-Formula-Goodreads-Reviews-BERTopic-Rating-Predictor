# Romance Novel Corpus Creation Pipeline

This module provides an automated pipeline for creating romance novel corpora from Anna's Archive datasets, designed for NLP research projects.

## 🆓 **FREE ALTERNATIVES AVAILABLE**

**NEW**: Free pipeline options that eliminate the need for API keys:

- **Torrent Datasets**: Access 1,000+ TB of books via torrent downloads
- **annas-mcp Tool**: Command-line tool for targeted searches
- **No API Key Required**: Completely free access to Anna's Archive content

See [Free Pipeline Usage](#free-pipeline-usage) section below for details.

## Overview

The corpus creation pipeline consists of three main phases:

1. **Book Matching**: Match Goodreads metadata with Anna's Archive entries using fuzzy matching
2. **Download**: Download matched books in preferred formats (epub > HTML > PDF)
3. **Storage**: Organize downloads with consistent naming and metadata tracking

## Features

- **Fuzzy Matching**: Intelligent matching using title, author, and publication year
- **Format Preference**: Prioritizes epub and HTML formats for ML processing
- **Error Handling**: Robust error handling with retry logic and detailed logging
- **Organized Storage**: Consistent file naming and directory structure
- **Scalable**: Batch processing for large datasets
- **Quality Tracking**: Comprehensive statistics and validation

## Quick Start

### Prerequisites

1. **Anna's Archive API Key**: Get an API key from [Anna's Archive](https://annas-archive.org/faq#api) (requires donation)
2. **Python Dependencies**: Install required packages
   ```bash
   pip install pandas requests beautifulsoup4
   ```

### Basic Usage

```python
from src.corpus_creation.pipeline import CorpusCreationPipeline, PipelineConfig
from pathlib import Path
import pandas as pd

# Load your Goodreads data
books_df = pd.read_csv('data/processed/romance_subdataset_6000.csv')

# Configure pipeline
config = PipelineConfig(
    api_key='your_api_key_here',
    storage_base=Path('./corpus_storage')
)

# Create and run pipeline
pipeline = CorpusCreationPipeline(config)
result = pipeline.run_full_pipeline(books_df)

# Check results
print(f"Processed {result.total_books_processed} books")
print(f"Found {result.matches_found} matches")
print(f"Downloaded {result.downloads_successful} books")
```

### Test with Sample Books

**Option 1: Free Pipeline (Recommended)**
```bash
cd src/corpus_creation
python test_free_pipeline.py --storage-dir ./free_corpus
```

**Option 2: API Pipeline (Requires API Key)**
```bash
cd src/corpus_creation
python test_pipeline.py --api-key YOUR_API_KEY --storage-dir ./test_corpus
```

## File Structure

```
src/corpus_creation/
├── __init__.py              # Package initialization
├── book_matcher.py          # Fuzzy matching algorithm
├── annas_client.py          # Anna's Archive API client
├── annas_torrent_client.py  # Free torrent dataset access
├── annas_mcp_client.py      # Free MCP tool integration
├── downloader.py            # Download and storage management
├── pipeline.py              # Main API-based pipeline
├── free_pipeline.py         # Free pipeline (no API key needed)
├── test_pipeline.py         # Test script for API pipeline
├── test_free_pipeline.py    # Test script for free pipeline
└── README.md               # This documentation
```

## Storage Structure

Downloaded books are organized as follows:

```
storage_base/
├── books/                   # Downloaded book files
│   ├── 846763_The_Duke_and_I_Julia_Quinn.epub
│   ├── 859012_Marriage_Most_Scandalous_Johanna_Lindsey.epub
│   └── ...
├── temp/                    # Temporary files during download
├── failed/                  # Failed download tracking
└── metadata/               # Pipeline results and statistics
    ├── match_results.csv   # Book matching results
    ├── download_results.csv # Download results
    └── pipeline_report.txt # Comprehensive report
```

## Configuration Options

### PipelineConfig Parameters

- `api_key`: Your Anna's Archive API key (required)
- `storage_base`: Base directory for storing downloads
- `min_confidence`: Minimum confidence threshold for matches (default: 0.7)
- `batch_size`: Number of books to process in each batch (default: 50)
- `max_retries`: Maximum download retries (default: 3)

### Format Preferences

The pipeline prioritizes formats in this order:
1. **epub** (preferred for ML processing)
2. **HTML** (good alternative)
3. **PDF** (fallback option)

## API Reference

### BookMatcher

Handles fuzzy matching between Goodreads and Anna's Archive data.

```python
matcher = BookMatcher(annas_client)
results = matcher.match_books_batch(books_df)
stats = matcher.get_match_statistics(results)
```

### AnnasArchiveClient

Client for Anna's Archive API interactions.

```python
client = AnnasArchiveClient(api_key)
books = client.search_books("Pride and Prejudice", "Jane Austen", 1813)
content = client.download_book(book_id, "epub")
```

### BookDownloader

Manages downloads and file organization.

```python
downloader = BookDownloader(storage_path, annas_client)
results = downloader.download_books_batch(match_results)
stats = downloader.get_storage_stats()
```

## Error Handling

The pipeline includes comprehensive error handling:

- **Network Issues**: Automatic retries with exponential backoff
- **Missing Books**: Flagged rather than failing the entire pipeline
- **Invalid Formats**: Graceful fallback to alternative formats
- **Storage Issues**: Clear error messages for permission/ space issues

## Logging

Detailed logs are written to `corpus_creation_test.log` and console. Log levels:
- **DEBUG**: Detailed execution information
- **INFO**: General progress and results
- **WARNING**: Non-critical issues
- **ERROR**: Critical failures

## Troubleshooting

### Common Issues

1. **Invalid API Key**: Ensure you have a valid API key from Anna's Archive
2. **Rate Limiting**: The pipeline includes automatic rate limiting
3. **Storage Permissions**: Ensure write access to the storage directory
4. **Network Issues**: Check internet connection and firewall settings

### Getting API Key

1. Visit [Anna's Archive FAQ](https://annas-archive.org/faq#api)
2. Look for API key information (may require donation)
3. Some repositories may provide test keys for development

## Performance Considerations

- **Batch Processing**: Process books in batches to manage memory usage
- **Rate Limiting**: Built-in delays to respect API limits
- **Parallel Downloads**: Can be extended for concurrent downloads
- **Storage Monitoring**: Track disk usage for large corpora

## Extending the Pipeline

The modular design makes it easy to extend:

- Add new format converters
- Implement additional matching algorithms
- Add quality validation steps
- Integrate with ML preprocessing pipelines

## Free Pipeline Usage

### 🆓 **No API Key Required!**

The free pipeline uses Anna's Archive torrent datasets and the annas-mcp tool:

```python
from src.corpus_creation.free_pipeline import FreeCorpusCreationPipeline, FreePipelineConfig
from pathlib import Path
import pandas as pd

# Load your Goodreads data
books_df = pd.read_csv('data/processed/romance_subdataset_6000.csv')

# Configure free pipeline
config = FreePipelineConfig(
    storage_base=Path('./free_corpus'),
    torrent_base=Path('./torrent_data'),
    use_torrents=True,
    use_mcp=True
)

# Create and run free pipeline
pipeline = FreeCorpusCreationPipeline(config)
result = pipeline.run_free_pipeline(books_df)

# Check results
print(f"Processed {result.total_books_processed} books")
print(f"Found {result.matches_found} matches")
print(f"Downloaded {result.downloads_successful} books")
```

### Available Free Datasets

Based on [Anna's Archive datasets](https://annas-archive.li/datasets):

- **Libgen.li**: 188TB - Fiction and non-fiction books
- **Libgen.rs**: 82TB - Scientific and technical books  
- **Z-Library**: 75TB - Academic and fiction books
- **Internet Archive**: 304TB - Public domain and digitized books
- **HathiTrust**: 9TB - Academic and research books
- **Sci-Hub**: 90TB - Scientific papers and books

### Installing annas-mcp Tool

1. Download from [GitHub Releases](https://github.com/iosifache/annas-mcp/releases)
2. Make executable: `chmod +x annas-mcp`
3. Set environment: `export ANNAS_DOWNLOAD_PATH="/path/to/downloads"`

## License and Ethics

- Ensure compliance with Anna's Archive terms of service
- Respect copyright and usage restrictions
- Use only for legitimate research purposes
- Consider data privacy and anonymization requirements

## Support

For issues or questions:
1. Check the logs for detailed error information
2. For free pipeline: Verify torrent client and annas-mcp installation
3. For API pipeline: Verify API key and network connectivity
4. Review the pipeline configuration
5. Check Anna's Archive status and documentation
