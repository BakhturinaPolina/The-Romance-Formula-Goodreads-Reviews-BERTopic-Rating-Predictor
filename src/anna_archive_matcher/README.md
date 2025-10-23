# Anna's Archive Full Automation System

## Overview
Fully automated system for finding and downloading romance books from Anna's Archive without manual intervention.

## Quick Start

### One-Command Full Automation
```bash
cd /home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research
source .venv/bin/activate
cd src/anna_archive_matcher

# Complete automation (100 top-rated books)
python fully_automated_workflow.py \
  --romance-csv ../../data/processed/romance_books_main_final_canonicalized.csv \
  --max-books 100 \
  --priority-list top_rated
```

### Quick Test
```bash
# Test with 20 books
python utils/robust_automated_search.py \
  --romance-csv utils/priority_lists/top_rated_popular_books.csv \
  --max-books 20 \
  --delay-min 1.0 \
  --delay-max 2.0
```

## Expected Results
- **Success rate**: 20-40% of books found automatically
- **Processing time**: 1-3 minutes per book
- **Quality**: High-quality matches with MD5 hashes

## Documentation
- `FULL_AUTOMATION_GUIDE.md` - Complete automation guide
- `README_FINAL.md` - Detailed solution summary

## Files
- `fully_automated_workflow.py` - Main automation workflow
- `utils/robust_automated_search.py` - Core automated search engine
- `utils/priority_lists/` - Pre-generated priority lists
