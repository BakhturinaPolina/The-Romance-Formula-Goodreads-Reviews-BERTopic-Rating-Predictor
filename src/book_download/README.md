# Anna's Archive Book Download & Title Matching

## Quick Start

### Direct Download (if you have MD5 hashes)
```bash
export ANNAS_SECRET_KEY="BvQiX8XK9Y2LNp5yuXXnV2yqyRPWP"
python3 anna_api_client.py "MD5_HASH" --download
```

### Title-to-MD5 Matching Pipeline (recommended)
```bash
# 1. Start Docker services
docker compose --env-file .env up -d

# 2. Ingest Anna's Archive data
./scripts/ingest_mariadb.sh data/maria/your_dump.sql.zst

# 3. Match titles to MD5s
python3 title_matcher_cli.py --backend mariadb --in data/your_titles.csv --out results.csv

# 4. Optional: Auto-download matched books
python3 title_matcher_cli.py --backend mariadb --in data/your_titles.csv --download --daily-limit 10
```

## Files
- `anna_api_client.py` - Direct API client for MD5-based downloads
- `title_matcher_cli.py` - **NEW**: Title-to-MD5 matching with dual backend support
- `mcp_integration.py` - MCP server integration
- `download_manager.py` - Batch processing with progress tracking
- `example_usage.py` - Usage examples
- `test_title_matcher.py` - Integration tests

## Key Features

### Title Matching Pipeline
- **Dual backends**: MariaDB and Elasticsearch support
- **Fuzzy matching**: Uses RapidFuzz for intelligent title/author matching
- **Confidence scoring**: Tiered matching (exact → high → medium → low confidence)
- **CSV integration**: Works with your existing book datasets
- **Auto-download**: Optional integration with existing download system

### Download System
- **Tor required** - Anna's Archive only accessible through Tor
- **MD5 needed** - Downloads require file MD5 hash, not title/author
- **Daily limit** - 25 downloads per day (configurable)
- **Progress tracking** - Resume interrupted downloads
- **Working** - Tested and functional with your API key

## Usage Examples

### Title Matching Only
```bash
# MariaDB backend
python3 title_matcher_cli.py --backend mariadb \
  --in data/processed/sample_50_books.csv \
  --out organized_outputs/anna_archive_download/title_matches.csv \
  --db-host localhost --db-name annas_archive --db-user annas_user --db-pass annas_pass

# Elasticsearch backend
python3 title_matcher_cli.py --backend es \
  --in data/processed/sample_50_books.csv \
  --out organized_outputs/anna_archive_download/title_matches.csv \
  --es-host http://localhost:9200 --index aa_records
```

### Title Matching + Auto Download
```bash
python3 title_matcher_cli.py --backend mariadb \
  --in data/processed/sample_50_books.csv \
  --download --daily-limit 10 \
  --db-host localhost --db-name annas_archive --db-user annas_user --db-pass annas_pass
```

### Direct Download (existing functionality)
```python
from anna_api_client import AnnaAPIClient
client = AnnaAPIClient()
result = client.download_book("MD5_HASH", "filename.epub")
```

## Docker Setup

The project includes a complete Docker Compose stack:

```bash
# Copy environment template
cp env.example .env

# Start services (MariaDB + Elasticsearch + Kibana)
docker compose --env-file .env up -d

# Check service status
docker compose ps
```

### Data Ingestion
```bash
# MariaDB: Load SQL dumps
./scripts/ingest_mariadb.sh data/maria/annas_archive_mariadb.sql.zst

# Elasticsearch: Load JSONL files
docker compose exec tools python scripts/ingest_es.py --index aa_records data/es/*.jsonl*
```

## Output Format

The title matcher produces CSV files with these columns:
- `work_id`, `input_title`, `input_author`, `input_year` (from your input)
- `md5`, `match_score`, `match_confidence` (matching results)
- `matched_title`, `matched_author`, `matched_year` (from Anna's Archive)
- `language`, `extension`, `filesize`, `isbn10`, `isbn13` (metadata)
- `timestamp` (processing time)

## Integration with Existing System

The title matcher integrates seamlessly with your existing download infrastructure:
- Uses the same output directory: `organized_outputs/anna_archive_download/`
- Leverages existing `BookDownloadManager` for batch downloads
- Respects daily download limits and progress tracking
- Maintains the same API key and Tor configuration
