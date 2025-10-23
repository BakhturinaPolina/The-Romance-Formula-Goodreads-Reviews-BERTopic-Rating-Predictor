# Step-by-Step Data Acquisition Guide

## Overview

This guide provides detailed step-by-step instructions for acquiring Anna's Archive data dumps and setting up the local search pipeline. Your API key `BvQiX8XK9Y2LNp5yuXXnV2yqyRPWP` will be used for downloading books after we find them in the data.

## Prerequisites

- **Disk Space**: At least 500GB free space (for full dataset) or 10GB (for samples)
- **RAM**: 8GB+ recommended (30GB+ for full dataset)
- **Internet**: Stable connection for downloading large files
- **Time**: 2-4 hours for full dataset download, 30 minutes for samples

## Step 1: Prepare Your System

### 1.1 Check Available Space
```bash
# Check available disk space
df -h

# You need at least 500GB for full dataset or 10GB for samples
```

### 1.2 Install Required Tools
```bash
# Install torrent client (if not already installed)
sudo apt update
sudo apt install qbittorrent-nox

# Or install GUI version
sudo apt install qbittorrent

# Install additional tools
sudo apt install wget curl
```

### 1.3 Verify Python Environment
```bash
# Navigate to your project
cd /home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research

# Activate virtual environment
source .venv/bin/activate

# Verify dependencies are installed
pip list | grep -E "(pandas|pyarrow|duckdb|ijson|requests|tqdm)"
```

## Step 2: Download Anna's Archive Data Dumps

### 2.1 Visit Anna's Archive Datasets Page

1. **Open your web browser**
2. **Go to**: https://annas-archive.org/datasets
3. **Look for the "Database Dumps" section**

### 2.2 Download Torrent Files

You'll need to download these torrent files:

#### **Primary Data (Required)**
- `elasticsearch.torrent` - Main book metadata (~200-300GB)
- `elasticsearchAux.torrent` - Auxiliary metadata (~50-100GB)

#### **Optional Data**
- `aac.torrent` - Combined data (~100-200GB)
- `mariadb.torrent` - Relational data (~50-100GB)

### 2.3 Download Using Torrent Client

#### **Option A: Command Line (qBittorrent-nox)**
```bash
# Start qBittorrent-nox in background
qbittorrent-nox -d

# Add torrents (replace with actual torrent file paths)
qbittorrent-nox --add-torrent=/path/to/elasticsearch.torrent --save-path=/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/data/anna_archive/elasticsearch/

# Check status
qbittorrent-nox --list
```

#### **Option B: GUI (qBittorrent)**
1. **Open qBittorrent**
2. **File → Add Torrent**
3. **Select the torrent files**
4. **Set save path to**: `/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/data/anna_archive/`
5. **Start downloads**

### 2.4 Monitor Download Progress

```bash
# Check download progress
ls -lh /home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/data/anna_archive/elasticsearch/

# You should see files like:
# part-00000.json.gz
# part-00001.json.gz
# part-00002.json.gz
# ...
```

**Expected download time**: 2-4 hours for full dataset (depending on connection speed)

## Step 3: Verify Downloaded Data

### 3.1 Check File Integrity
```bash
# Navigate to data directory
cd /home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/data/anna_archive/elasticsearch/

# List files and sizes
ls -lh *.json.gz

# Test decompression of first file
gunzip -t part-00000.json.gz

# If successful, you'll see no output (no errors)
```

### 3.2 Quick Data Inspection
```bash
# Look at the structure of the first few records
gunzip -c part-00000.json.gz | head -5

# You should see JSON records like:
# {"_source": {"file_unified_data": {"title": {"best": "Book Title"}, "author": {"best": "Author Name"}, ...}}}
```

## Step 4: Create Sample Data (Recommended First Step)

### 4.1 Extract Small Sample for Testing
```bash
# Navigate to the local pipeline directory
cd /home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/src/anna_archive_local

# Extract 10,000 records for development
python3 sample_data_extractor.py \
  --input-dir ../../data/anna_archive/elasticsearch/ \
  --output-file ../../data/anna_archive/elasticsearch/sample_10k.json.gz \
  --sample-size 10000 \
  --analyze

# This will create a sample file and show statistics
```

### 4.2 Convert Sample to Parquet
```bash
# Convert the sample to Parquet format
python3 json_to_parquet.py \
  --input-file ../../data/anna_archive/elasticsearch/sample_10k.json.gz \
  --output-dir ../../data/anna_archive/parquet/sample_10k/

# This will create Parquet files in the parquet directory
```

## Step 5: Test the Pipeline

### 5.1 Test Search Engine
```bash
# Test the search engine with sample data
python3 book_search_cli.py \
  --parquet-dir ../../data/anna_archive/parquet/sample_10k/ \
  --stats

# You should see dataset statistics like:
# Total records: 10,000
# Schema fields: 8
# Field coverage: title (95%), author (90%), etc.
```

### 5.2 Test Book Search
```bash
# Search for a specific book
python3 book_search_cli.py \
  --parquet-dir ../../data/anna_archive/parquet/sample_10k/ \
  --title "Romeo and Juliet" \
  --author "Shakespeare"

# Try searching for books from your sample_50_books.csv
python3 book_search_cli.py \
  --parquet-dir ../../data/anna_archive/parquet/sample_10k/ \
  --title "Fifty Shades" \
  --author "E.L. James"
```

### 5.3 Test API Downloader
```bash
# Test API connection with your key
python3 api_downloader.py \
  --api-key "BvQiX8XK9Y2LNp5yuXXnV2yqyRPWP" \
  --test

# You should see: "✅ API connection successful"
```

## Step 6: Run Complete Demo

### 6.1 Test with 50 Sample Books
```bash
# Run the complete demo
python3 demo_query_50_books.py \
  --parquet-dir ../../data/anna_archive/parquet/sample_10k/ \
  --books-csv ../../data/processed/sample_50_books.csv \
  --output-dir ../../data/anna_archive/demo_results/ \
  --api-key "BvQiX8XK9Y2LNp5yuXXnV2yqyRPWP"

# This will:
# 1. Search for all 50 books in your CSV
# 2. Generate detailed results
# 3. Test downloading a few books
# 4. Create comprehensive reports
```

### 6.2 Review Results
```bash
# Check the demo results
ls -la ../../data/anna_archive/demo_results/

# You should see:
# - search_results.csv (detailed results)
# - search_summary.json (statistics)
# - download_ready.csv (books with MD5 hashes)
# - search_report.md (human-readable report)
# - test_downloads/ (sample downloaded books)
```

## Step 7: Scale Up to Full Dataset (Optional)

### 7.1 Convert Full Dataset
```bash
# Only do this if you have the full dataset and sufficient resources
# This will take 4-8 hours and require 30GB+ RAM

# Convert all JSON files to Parquet
python3 json_to_parquet.py \
  --input-dir ../../data/anna_archive/elasticsearch/ \
  --output-dir ../../data/anna_archive/parquet/full/ \
  --chunk-size 20000 \
  --sample-size 5000
```

### 7.2 Test Full Dataset
```bash
# Test with full dataset
python3 book_search_cli.py \
  --parquet-dir ../../data/anna_archive/parquet/full/ \
  --stats

# Run demo with full dataset
python3 demo_query_50_books.py \
  --parquet-dir ../../data/anna_archive/parquet/full/ \
  --books-csv ../../data/processed/sample_50_books.csv \
  --output-dir ../../data/anna_archive/demo_results_full/ \
  --api-key "BvQiX8XK9Y2LNp5yuXXnV2yqyRPWP"
```

## Step 8: Download Books Using API

### 8.1 Download Individual Books
```bash
# Download a specific book by MD5
python3 api_downloader.py \
  --api-key "BvQiX8XK9Y2LNp5yuXXnV2yqyRPWP" \
  --md5 "8336332bf5877e3adbfb60ac70720cd5" \
  --output-dir ../../organized_outputs/epub_downloads/
```

### 8.2 Batch Download
```bash
# Download all books from demo results
python3 api_downloader.py \
  --api-key "BvQiX8XK9Y2LNp5yuXXnV2yqyRPWP" \
  --csv-file ../../data/anna_archive/demo_results/download_ready.csv \
  --output-dir ../../organized_outputs/epub_downloads/ \
  --delay 2.0
```

## Troubleshooting

### Common Issues and Solutions

#### **Issue 1: "No torrent files found"**
```bash
# Check if you're on the right page
# Go to: https://annas-archive.org/datasets
# Look for "Database Dumps" section
# Download torrent files to your local machine first
```

#### **Issue 2: "Insufficient disk space"**
```bash
# Check available space
df -h

# Free up space or use external storage
# Consider using samples instead of full dataset
```

#### **Issue 3: "Download too slow"**
```bash
# Use torrent client with multiple connections
# Download during off-peak hours
# Consider downloading only essential files first
```

#### **Issue 4: "API connection failed"**
```bash
# Verify your API key is correct
python3 api_downloader.py --api-key "BvQiX8XK9Y2LNp5yuXXnV2yqyRPWP" --test

# Check network connectivity
ping annas-archive.org
```

#### **Issue 5: "No books found in search"**
```bash
# Try with larger sample size
python3 sample_data_extractor.py --sample-size 50000

# Use fuzzy matching (default)
python3 book_search_cli.py --title "partial title" --author "partial author"

# Check if your sample contains the books you're searching for
```

## Expected Results

### Sample Dataset (10K records)
- **Processing time**: 10-20 minutes
- **Disk usage**: ~500MB
- **Match rate**: 10-30% for romance books
- **Query speed**: <5ms

### Full Dataset (~10M records)
- **Processing time**: 4-8 hours
- **Disk usage**: ~500GB
- **Match rate**: 30-50% for romance books
- **Query speed**: <100ms

## Next Steps

1. **Start with samples**: Always test with small samples first
2. **Monitor performance**: Watch for memory/disk issues
3. **Scale gradually**: Increase sample size as you validate the pipeline
4. **Integrate with research**: Use the downloaded books for your NLP research
5. **Compare approaches**: Test both local pipeline and web scraping methods

## Support

If you encounter issues:

1. **Check logs**: All scripts provide detailed logging
2. **Use verbose mode**: Add `--verbose` to any command
3. **Start small**: Use 1K samples for initial testing
4. **Check documentation**: Refer to README.md and other guides
5. **Verify prerequisites**: Ensure sufficient disk space and RAM

## Summary

This pipeline provides a powerful alternative to web scraping for Anna's Archive book searches. With your API key `BvQiX8XK9Y2LNp5yuXXnV2yqyRPWP`, you can:

- Search millions of books offline
- Download books directly using MD5 hashes
- Process large datasets efficiently
- Avoid rate limiting and network issues

Start with the sample approach, validate everything works, then scale up to the full dataset when ready!
