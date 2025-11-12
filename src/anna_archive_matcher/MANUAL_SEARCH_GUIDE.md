# Manual Search Guide for Romance Books

## Overview

Since Anna's Archive full datasets require torrent downloads, here are practical alternatives for finding romance books without torrents.

## Strategy 1: Targeted Anna's Archive Search

### Step 1: Use Anna's Archive Search Interface
1. Go to [https://annas-archive.org](https://annas-archive.org)
2. Use the search bar with specific queries
3. Filter results by:
   - **Language**: English
   - **Content Type**: Books
   - **File Format**: EPUB, PDF
   - **Genre**: Romance, Fiction

### Step 2: Search Queries for Romance Books
```
# Popular romance authors
"Nicholas Sparks" romance
"E.L. James" fiction
"Jamie McGuire" romance
"Kiera Cass" young adult romance

# Romance subgenres
"contemporary romance" English
"historical romance" fiction
"paranormal romance" English
"chick lit" English

# Popular romance series
"Fifty Shades" trilogy
"Beautiful" series romance
"Selection" series young adult
```

### Step 3: Download Individual Books
- Click on book titles to access download pages
- Look for direct download links (not torrents)
- Download in EPUB or PDF format
- Note the MD5 hash for your records

## Strategy 2: Alternative Free Sources

### Project Gutenberg
- **URL**: [https://www.gutenberg.org](https://www.gutenberg.org)
- **Focus**: Public domain romance classics
- **Advantage**: Direct downloads, no restrictions
- **Limitation**: Older books (pre-1920s)

### Internet Archive
- **URL**: [https://archive.org](https://archive.org)
- **Focus**: Digitized books, including some romance
- **Advantage**: Free access, multiple formats
- **Search**: Use advanced search with "romance" + "fiction"

### Open Library
- **URL**: [https://openlibrary.org](https://openlibrary.org)
- **Focus**: Book metadata and some free downloads
- **Advantage**: Good for finding book information
- **Limitation**: Limited free downloads

## Strategy 3: Automated Search Script

Use the provided scripts to automate the search process:

### Test Alternative Sources
```bash
cd src/anna_archive_matcher/utils
python alternative_sources.py \
  --romance-csv ../../data/processed/romance_books_main_final_canonicalized.csv \
  --test-search
```

### Create Alternative Dataset
```bash
python alternative_sources.py \
  --romance-csv ../../data/processed/romance_books_main_final_canonicalized.csv \
  --output-csv alternative_romance_books.csv
```

## Strategy 4: Focused Download Approach

### Priority Books for Download
Focus on the most popular books from your dataset:

1. **Top-rated books** (average_rating_weighted_mean > 4.0)
2. **Most reviewed books** (ratings_count_sum > 100,000)
3. **Recent publications** (publication_year > 2010)
4. **Popular authors** (multiple books by same author)

### Sample Priority List
Based on your dataset, prioritize:
- Fifty Shades series (E.L. James)
- Beautiful series (Jamie McGuire)
- Selection series (Kiera Cass)
- Nicholas Sparks novels
- Popular contemporary romance authors

## Strategy 5: Batch Processing Workflow

### Step 1: Create Priority List
```python
import pandas as pd

# Load your romance dataset
df = pd.read_csv('romance_books_main_final_canonicalized.csv')

# Create priority list
priority_books = df[
    (df['average_rating_weighted_mean'] > 4.0) & 
    (df['ratings_count_sum'] > 50000)
].head(100)

priority_books.to_csv('priority_romance_books.csv', index=False)
```

### Step 2: Manual Search and Download
1. Open `priority_romance_books.csv`
2. Search each book manually on Anna's Archive
3. Download available books
4. Record MD5 hashes in a spreadsheet

### Step 3: Batch Download
Use your existing download system with the collected MD5 hashes:
```bash
python standalone_downloader.py \
  --csv collected_md5_hashes.csv \
  --output-dir organized_outputs/epub_downloads
```

## Expected Results

### Realistic Expectations
- **Manual search**: 10-20% of books found
- **Alternative sources**: 5-15% of books found
- **Combined approach**: 15-30% of books found
- **Time investment**: 2-4 hours for 100 priority books

### Quality vs Quantity
- **High-quality matches**: Focus on exact title/author matches
- **Acceptable matches**: Same book, different edition
- **Avoid**: Books with significantly different titles/authors

## Tools and Resources

### Browser Extensions
- **Anna's Archive Helper**: Automates some search tasks
- **Download Manager**: Helps organize downloads

### Spreadsheet Templates
Create a tracking spreadsheet with columns:
- Original Title
- Original Author
- Found Title
- Found Author
- MD5 Hash
- Download URL
- Status (Found/Downloaded/Failed)

### Automation Scripts
Use the provided Python scripts to:
- Search multiple sources automatically
- Track download progress
- Generate reports

## Legal and Ethical Considerations

### Important Notes
- **Respect copyright**: Only download books you have rights to access
- **Rate limiting**: Don't overwhelm servers with requests
- **Terms of service**: Follow each platform's usage guidelines
- **Personal use**: Ensure downloads are for personal research use

### Best Practices
- Use official sources when possible
- Support authors by purchasing books when available
- Use downloads for research purposes only
- Respect platform terms of service

## Next Steps

1. **Start with priority books**: Focus on top-rated, popular books
2. **Test the workflow**: Try manual search for 10-20 books first
3. **Use automation**: Implement the provided scripts for efficiency
4. **Track progress**: Maintain a spreadsheet of found/downloaded books
5. **Scale gradually**: Expand to more books as the process becomes efficient

This approach will give you a substantial collection of romance books without requiring torrent downloads or full dataset processing.
