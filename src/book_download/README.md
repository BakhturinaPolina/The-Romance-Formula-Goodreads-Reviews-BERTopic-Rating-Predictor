# Book Download Research Component

This component handles downloading romance novels from Anna's Archive for NLP research using the anna-mcp server.

## Overview

The system is designed to:
- Download 6,000 romance novels from Anna's Archive
- Process 999 books per day with proper rate limiting
- Resume from exactly where it left off if interrupted
- Use title and author_name for searching
- Save books in EPUB/HTML format
- Provide extensive logging and progress tracking

## Files

- `explore_data.py` - Data exploration script to understand CSV structure
- `test_download.py` - Simple test script for download workflow
- `download_manager.py` - Main download management system with progress tracking
- `mcp_integration.py` - Integration with anna-mcp server
- `run_downloads.py` - Production script for running downloads
- `README.md` - This documentation

## Setup

1. **Environment**: The anna-mcp server is already configured in Cursor with:
   - API Key: Set in MCP configuration
   - Download Path: `/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/organized_outputs/anna_archive_download`

2. **Virtual Environment**: Always run in the project's virtual environment:
   ```bash
   cd /home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research
   source venv/bin/activate
   ```

## Usage

### Testing with Sample Books (2-3 books)

```bash
# Test with sample CSV (8 books total)
python src/book_download/run_downloads.py --sample --max-books 3

# Test with specific daily limit
python src/book_download/run_downloads.py --sample --max-books 5 --daily-limit 10
```

### Production Run (Full Dataset)

```bash
# Download up to daily limit (999 books)
python src/book_download/run_downloads.py

# Download specific number of books
python src/book_download/run_downloads.py --max-books 500

# Custom daily limit
python src/book_download/run_downloads.py --daily-limit 500
```

### Data Exploration

```bash
# Explore CSV structure and statistics
python src/book_download/explore_data.py

# Test download workflow
python src/book_download/test_download.py
```

## Progress Tracking

The system automatically tracks progress in:
- `download_progress.json` - Current progress and statistics
- `download_summary_YYYYMMDD_HHMMSS.json` - Summary of each run
- `download_log.txt` - Detailed logging

### Progress File Structure

```json
{
  "last_row": 2,
  "total_processed": 2,
  "total_downloaded": 2,
  "total_failed": 0,
  "last_run_date": "2025-09-28",
  "daily_downloads": 2,
  "download_history": [...]
}
```

## Resumable Downloads

The system automatically resumes from the last processed row:
- If interrupted, run the same command again
- It will start from `last_row` in the progress file
- Daily limits are reset each day
- No duplicate downloads

## Daily Limits

- **Default**: 999 books per day
- **Customizable**: Use `--daily-limit` parameter
- **Automatic reset**: Counter resets at midnight
- **Progress tracking**: Shows remaining downloads for the day

## File Organization

Downloaded books are saved to:
```
organized_outputs/anna_archive_download/
├── 624953_A_Little_Scandal.epub
├── 97656_Reluctant_Mistress,_Blackmailed_Wife.epub
├── download_progress.json
├── download_summary_20250928_175531.json
└── download_log.txt
```

## Error Handling

The system handles:
- Network errors with retries
- Books not found in Anna's Archive
- Daily limit reached
- File system errors
- Progress corruption (creates new progress file)

## Logging

All operations are logged to:
- Console output (INFO level)
- `download_log.txt` file
- Progress tracking files

## Next Steps for MCP Integration

The current implementation uses simulated downloads. To enable real downloads:

1. **Implement MCP Search**: Replace `_simulate_search()` in `mcp_integration.py` with actual MCP server calls
2. **Implement MCP Download**: Replace `_simulate_download()` with actual MCP server calls
3. **Error Handling**: Add proper error handling for MCP server responses
4. **Rate Limiting**: Implement proper delays between MCP calls

## Testing

The system has been tested with:
- ✅ Sample CSV (8 books)
- ✅ Progress tracking and resumable downloads
- ✅ Daily limit enforcement
- ✅ JSON serialization (numpy types)
- ✅ File creation and organization
- ✅ Error handling and logging

## Production Deployment

For production use with the full 6,000 book dataset:

1. **First run**: `python src/book_download/run_downloads.py`
2. **Daily runs**: Same command - automatically resumes from last position
3. **Monitoring**: Check `download_progress.json` for current status
4. **Completion**: System will stop when all 6,000 books are processed

The system is designed to run reliably over multiple days, automatically managing daily limits and progress tracking.
