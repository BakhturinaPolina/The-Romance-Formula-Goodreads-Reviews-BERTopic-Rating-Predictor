# Title-to-MD5 Matching Pipeline

## Overview

This pipeline maps book titles from your CSV datasets to Anna's Archive MD5 hashes, enabling automatic book downloads. It supports both MariaDB and Elasticsearch backends and integrates seamlessly with the existing download system.

## Architecture

```
CSV Input → Title Matcher → MD5 Output → Download Manager → Books
    ↓           ↓              ↓              ↓
  titles    fuzzy match    confidence    auto download
  authors   scoring        filtering     progress tracking
  years     tiered rules   validation    daily limits
```

## Components

### 1. Docker Infrastructure
- **MariaDB**: Relational database for Anna's Archive metadata
- **Elasticsearch**: Full-text search engine for fuzzy matching
- **Kibana**: Web interface for data exploration
- **Tools Container**: Python environment with all dependencies

### 2. Title Matcher CLI (`title_matcher_cli.py`)
- Dual backend support (MariaDB + Elasticsearch)
- Fuzzy matching with RapidFuzz
- Confidence scoring and tiered filtering
- CSV input/output processing
- Optional auto-download integration

### 3. Data Ingestion Scripts
- `scripts/ingest_mariadb.sh`: Load SQL dumps into MariaDB
- `scripts/ingest_es.py`: Load JSONL files into Elasticsearch

### 4. Integration Layer
- Uses existing `BookDownloadManager` for downloads
- Maintains progress tracking and daily limits
- Outputs to `organized_outputs/anna_archive_download/`

## Setup Instructions

### 1. Environment Setup
```bash
# Copy environment template
cp env.example .env

# Edit .env with your settings
nano .env
```

### 2. Start Docker Services
```bash
# Start all services
docker compose --env-file .env up -d

# Check service status
docker compose ps

# View logs if needed
docker compose logs mariadb
docker compose logs elasticsearch
```

### 3. Data Ingestion

#### MariaDB (Recommended for smaller datasets)
```bash
# Place your SQL dump in data/maria/
# Example: annas_archive_mariadb.sql.zst

# Ingest the data
./scripts/ingest_mariadb.sh data/maria/annas_archive_mariadb.sql.zst

# Verify ingestion
docker compose exec mariadb mysql -u annas_user -pannas_pass annas_archive -e "SELECT COUNT(*) FROM aa_records;"
```

#### Elasticsearch (Recommended for large datasets)
```bash
# Place your JSONL files in data/es/
# Example: annas_archive_*.jsonl.zst

# Ingest the data
docker compose exec tools python scripts/ingest_es.py --index aa_records --create-index data/es/*.jsonl*

# Verify ingestion
curl http://localhost:9200/aa_records/_count
```

## Usage Examples

### Basic Title Matching
```bash
# MariaDB backend
python3.11 src/book_download/title_matcher_cli.py \
  --backend mariadb \
  --in data/processed/sample_50_books.csv \
  --out organized_outputs/anna_archive_download/title_matches.csv \
  --db-host localhost --db-name annas_archive --db-user annas_user --db-pass annas_pass

# Elasticsearch backend
python3.11 src/book_download/title_matcher_cli.py \
  --backend es \
  --in data/processed/sample_50_books.csv \
  --out organized_outputs/anna_archive_download/title_matches.csv \
  --es-host http://localhost:9200 --index aa_records
```

### Title Matching + Auto Download
```bash
python3.11 src/book_download/title_matcher_cli.py \
  --backend mariadb \
  --in data/processed/sample_50_books.csv \
  --download --daily-limit 10 \
  --db-host localhost --db-name annas_archive --db-user annas_user --db-pass annas_pass
```

### Using Docker Tools Container
```bash
# Run inside the tools container
docker compose exec tools python src/book_download/title_matcher_cli.py \
  --backend mariadb \
  --in data/processed/sample_50_books.csv \
  --out organized_outputs/anna_archive_download/title_matches.csv
```

## Matching Algorithm

### 1. Input Processing
- Normalizes titles and authors using Unidecode
- Handles missing or malformed data gracefully
- Extracts publication years for additional matching criteria

### 2. Backend Search
- **MariaDB**: Uses LIKE queries with fuzzy scoring
- **Elasticsearch**: Uses match queries with fuzziness AUTO

### 3. Scoring System
```python
total_score = title_score * 0.75 + author_score * 0.2 + year_score * 0.05
```

### 4. Confidence Tiers
- **exact_match**: title ≥ 95% AND author ≥ 80%
- **high_confidence**: title ≥ 90% AND author ≥ 70%
- **medium_confidence**: title ≥ 80%
- **low_confidence**: title < 80%
- **no_matches**: No candidates found

### 5. Output Filtering
- Returns best match per input title
- Includes confidence score and explanation
- Provides metadata (language, extension, file size, ISBN)

## Output Format

The title matcher produces CSV files with these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `work_id` | Original work ID | 12345 |
| `input_title` | Input title | "Pride and Prejudice" |
| `input_author` | Input author | "Jane Austen" |
| `input_year` | Input publication year | 1813 |
| `md5` | Matched MD5 hash | "a1b2c3d4e5f6..." |
| `match_score` | Confidence score (0-100) | 95.2 |
| `match_confidence` | Confidence tier | "exact_match" |
| `matched_title` | Matched title from AA | "Pride and Prejudice" |
| `matched_author` | Matched author from AA | "Austen, Jane" |
| `matched_year` | Matched year from AA | 1813 |
| `language` | Book language | "en" |
| `extension` | File extension | "epub" |
| `filesize` | File size in bytes | 1024000 |
| `isbn10` | ISBN-10 | "0141439513" |
| `isbn13` | ISBN-13 | "9780141439518" |
| `timestamp` | Processing timestamp | "2025-01-21T22:38:24" |

## Integration with Download System

### Automatic Downloads
When using the `--download` flag, the title matcher:
1. Processes titles and generates MD5 mappings
2. Creates a temporary CSV with MD5 hashes
3. Calls `BookDownloadManager.run_md5_download_batch()`
4. Respects daily download limits
5. Maintains progress tracking
6. Saves files to `organized_outputs/anna_archive_download/`

### Progress Tracking
- Uses existing `download_progress.json` system
- Tracks daily download counts
- Resumes interrupted batches
- Provides detailed logging

## Performance Considerations

### MariaDB
- **Pros**: Simple setup, easy SQL queries, good for smaller datasets
- **Cons**: Slower fuzzy matching, requires more manual optimization
- **Recommended for**: < 1M records, development/testing

### Elasticsearch
- **Pros**: Fast fuzzy search, built-in full-text capabilities, scales well
- **Cons**: Higher memory usage, more complex setup
- **Recommended for**: > 1M records, production use

### Optimization Tips
- Use appropriate batch sizes (1000-5000 records)
- Index frequently searched columns (title, author)
- Monitor memory usage during large operations
- Use connection pooling for high-volume processing

## Troubleshooting

### Common Issues

#### 1. Connection Errors
```bash
# Check service status
docker compose ps

# Check logs
docker compose logs mariadb
docker compose logs elasticsearch

# Test connections
docker compose exec mariadb mysql -u annas_user -pannas_pass annas_archive -e "SELECT 1;"
curl http://localhost:9200/_cluster/health
```

#### 2. No Matches Found
- Verify data ingestion completed successfully
- Check table/index names match configuration
- Ensure input CSV has correct column names
- Try with more permissive fuzzy matching

#### 3. Low Match Quality
- Review confidence thresholds
- Check for data quality issues in input CSV
- Consider preprocessing titles (remove extra spaces, normalize punctuation)
- Use ISBN matching for higher precision

#### 4. Download Failures
- Verify API key is set: `echo $ANNAS_SECRET_KEY`
- Check Tor connectivity: `torsocks curl -s https://annas-archive.se`
- Monitor daily download limits
- Check disk space in download directory

### Debug Mode
```bash
# Enable verbose logging
python3.11 src/book_download/title_matcher_cli.py \
  --backend mariadb \
  --in data/processed/sample_50_books.csv \
  --verbose \
  --db-host localhost --db-name annas_archive --db-user annas_user --db-pass annas_pass
```

## Data Sources

### Anna's Archive Datasets
- **Official datasets**: Available via torrent from Anna's Archive
- **Typical sizes**: MariaDB ~500GB, Elasticsearch ~120GB + 600GB aux
- **Update frequency**: Varies, check official announcements
- **Legal considerations**: Ensure compliance with local laws

### Alternative Sources
- **Z-Library datasets**: Available through Anna's Archive
- **Internet Archive**: Public domain books
- **Project Gutenberg**: Free ebooks
- **Open Library**: Open bibliographic data

## Security Considerations

### API Key Management
- Store API key in environment variables
- Never commit API keys to version control
- Use `.env` files for local development
- Rotate keys periodically

### Tor Configuration
- Ensure Tor is properly configured
- Use torsocks for all Anna's Archive connections
- Monitor for connection issues
- Consider using Tor bridges if needed

### Data Privacy
- Process data locally when possible
- Be mindful of copyright restrictions
- Anonymize any personal data in logs
- Follow academic use guidelines

## Future Enhancements

### Planned Features
- **Batch processing**: Process multiple CSV files
- **Incremental updates**: Handle new data without full reprocessing
- **Quality metrics**: Detailed matching statistics
- **Web interface**: Browser-based matching interface
- **API endpoints**: REST API for programmatic access

### Integration Opportunities
- **Goodreads API**: Cross-reference with Goodreads data
- **ISBN lookup**: Enhanced matching using ISBN databases
- **Language detection**: Automatic language identification
- **Genre classification**: Book genre matching and filtering
