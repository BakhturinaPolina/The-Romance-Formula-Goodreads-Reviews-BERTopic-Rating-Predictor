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

- `run_downloads.py` - Main production script for running downloads
- `download_manager_production.py` - Production download management system with progress tracking
- `mcp_integration_final.py` - Production-ready integration with anna-mcp server
- `FORMAT_ISSUE_ANALYSIS.md` - Detailed analysis of format detection issue and solution
- `PRODUCTION_READY_SUMMARY.md` - Complete production readiness summary
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

### Format Detection

The system automatically detects and handles different file formats:
- **Format Detection**: Uses `file` command to detect actual format (EPUB, MOBI, PDF, etc.)
- **Automatic Renaming**: Files renamed to reflect actual format
- **Format Validation**: Ensures correct file extensions match content

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
├── 624953_A_Little_Scandal.mobi          # MOBI format (auto-detected)
├── 97656_Reluctant_Mistress_Blackmailed_Wife.epub  # EPUB format
├── download_progress.json                # Progress tracking
├── download_summary_YYYYMMDD_HHMMSS.json # Run summaries
└── download_log.txt                      # Detailed logging
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

## Production Status

The system is **PRODUCTION READY** with:
- ✅ **Real MCP Integration**: Working with actual Anna's Archive
- ✅ **Format Detection**: Automatic format detection and renaming
- ✅ **Error Handling**: Robust error handling for all scenarios
- ✅ **Progress Tracking**: Resumable downloads with progress tracking
- ✅ **Daily Limits**: 999 books per day with automatic reset

## Testing

The system has been thoroughly tested with:
- ✅ **MCP Integration**: Real MCP search and download functionality
- ✅ **Format Detection**: Correctly detects MOBI vs EPUB formats
- ✅ **Progress Tracking**: Resumable downloads working
- ✅ **Error Handling**: Proper error handling for all scenarios
- ✅ **Production Workflow**: Complete end-to-end functionality
- ✅ **Sample Data**: Tested with 8-book sample dataset

## Production Deployment

For production use with the full 6,000 book dataset:

1. **First run**: `python src/book_download/run_downloads.py`
2. **Daily runs**: Same command - automatically resumes from last position
3. **Monitoring**: Check `download_progress.json` for current status
4. **Completion**: System will stop when all 6,000 books are processed

The system is designed to run reliably over multiple days, automatically managing daily limits and progress tracking.
