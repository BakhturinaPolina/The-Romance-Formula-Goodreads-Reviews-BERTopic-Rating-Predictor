# Anna's Archive Book Matcher - Setup Guide

This guide will help you set up the automated book matching system to find your romance novels in Anna's Archive datasets and extract MD5 hashes for batch downloading.

## Overview

The system uses DuckDB to perform high-performance SQL queries on Anna's Archive datasets, automatically matching your romance books and extracting MD5 hashes for download.

## Prerequisites

- Python 3.8+
- 30GB+ RAM (for processing large datasets)
- 100GB+ storage space (for Anna Archive datasets)
- Virtual environment activated

## Step 1: Install Dependencies

```bash
cd /home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/src/anna_archive_matcher
pip install -r requirements.txt
```

## Step 2: Download Anna's Archive Datasets

### Option A: Manual Download (Recommended)

1. Visit [Anna's Archive Datasets](https://annas-archive.li/datasets)
2. Download the following datasets:
   - **aarecord_elasticsearch** (Elasticsearch dataset)
   - **aarecord_aac** (AAC dataset) 
   - **aarecord_mariadb** (MariaDB dataset)

3. Extract and place files in the appropriate directories:
   ```
   data/
   ├── elasticsearch/    # Place .gz files here
   ├── aac/             # Place .zst files here
   └── mariadb/         # Place .gz files here
   ```

### Option B: Automated Setup

```bash
python setup_datasets.py --data-dir data
```

## Step 3: Process Raw Data Files

Convert Anna's Archive raw files to Parquet format for efficient querying:

```bash
python run_matcher.py --process-data --data-dir data
```

This will:
- Decompress .gz and .zst files
- Convert JSON to Parquet format
- Create optimized datasets in `data/elasticsearchF/`, `data/aacF/`, `data/mariadbF/`

## Step 4: Run Book Matching

Match your romance books with Anna's Archive datasets:

```bash
python run_matcher.py \
  --romance-csv ../../data/processed/romance_books_main_final_canonicalized.csv \
  --data-dir data \
  --output-dir outputs \
  --similarity-threshold 0.8
```

### Parameters:
- `--romance-csv`: Path to your romance books CSV
- `--data-dir`: Path to Anna Archive data directory
- `--output-dir`: Directory for output files
- `--similarity-threshold`: Matching threshold (0.0-1.0)
- `--sample-size`: Process only N books (for testing)

## Step 5: Review Results

The system generates several output files:

- `outputs/anna_archive_matches.csv`: All matches found
- `outputs/download_ready_books.csv`: Books with MD5 hashes ready for download
- `outputs/matching_summary_report.txt`: Detailed summary report

## Step 6: Batch Download Books

Use your existing download system with the generated MD5 hashes:

```bash
cd /home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research
python standalone_downloader.py \
  --csv outputs/download_ready_books.csv \
  --output-dir organized_outputs/epub_downloads
```

## Interactive Analysis

For detailed analysis and visualization, use the Jupyter notebook:

```bash
jupyter notebook notebooks/anna_archive_analysis.ipynb
```

## Expected Results

Based on the Anna's Archive datasets, you can expect:

- **Match Rate**: 20-40% of romance books (depending on dataset coverage)
- **Download Rate**: 15-30% of books (those with valid MD5 hashes)
- **File Types**: Primarily EPUB, PDF, and other ebook formats
- **Sources**: Mix of Elasticsearch, AAC, and MariaDB datasets

## Troubleshooting

### Common Issues:

1. **"No matches found"**
   - Check if Anna Archive datasets are properly processed
   - Verify data directory structure
   - Try lowering similarity threshold

2. **Memory errors**
   - Reduce chunk size in configuration
   - Process smaller samples first
   - Ensure sufficient RAM (30GB+)

3. **Slow processing**
   - Increase DuckDB memory limit
   - Use SSD storage for better I/O
   - Process in smaller batches

### Performance Tips:

- Use SSD storage for better performance
- Process samples first to test configuration
- Adjust DuckDB memory settings based on your system
- Use parallel processing where possible

## File Structure

```
anna_archive_matcher/
├── data/
│   ├── elasticsearch/    # Raw .gz files
│   ├── elasticsearchF/   # Processed .parquet files
│   ├── aac/             # Raw .zst files
│   ├── aacF/            # Processed .parquet files
│   ├── mariadb/         # Raw .gz files
│   └── mariadbF/        # Processed .parquet files
├── outputs/             # Generated results
├── notebooks/           # Analysis notebooks
├── core/               # Core matching logic
├── utils/              # Utility functions
└── requirements.txt    # Dependencies
```

## Next Steps

1. **Test with Sample**: Start with `--sample-size 100` to test the system
2. **Optimize Parameters**: Adjust similarity threshold based on results
3. **Scale Up**: Process full dataset once configuration is optimized
4. **Integrate**: Use results with your existing download system

## Support

For issues or questions:
- Check the logs in `anna_archive_matcher.log`
- Review the summary report for detailed statistics
- Use the interactive notebook for debugging
- Refer to the [Anna's Archive Data Science Starter Kit](https://github.com/RArtutos/Data-science-starter-kit-Enhance/) for reference
