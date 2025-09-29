# Book Download Research Component

A robust system for downloading romance novels from Anna's Archive with automatic format validation, conversion, and progress tracking.

## Overview

This component provides a complete solution for downloading books from Anna's Archive using IPFS gateways, with built-in EPUB validation, MOBI-to-EPUB conversion via Calibre, and comprehensive error handling.

## Features

- **🔍 Smart Search**: Search Anna's Archive using title and author
- **📥 Robust Downloads**: IPFS-based downloads with multiple gateway fallbacks
- **🛡️ EPUB Guard**: Automatic format detection and validation
- **🔄 Format Conversion**: MOBI to EPUB conversion via Calibre
- **📊 Progress Tracking**: Resume-capable download sessions with daily limits
- **🔧 File Repair**: Fix mislabeled files in existing download directories
- **⚡ Hardened IPFS**: User-Agent headers and exponential backoff for 403 errors

## Components

### Core Files

- **`download_manager.py`** - Main orchestrator for batch downloads
- **`mcp_integration.py`** - Integration with anna-mcp server
- **`aa_epub_guard.py`** - EPUB validation and format conversion
- **`repair_existing_files.py`** - One-shot repair for mislabeled files

### Configuration Files

- **`EPUB_GUARD_INTEGRATION.md`** - Technical documentation for EPUB guard

## Prerequisites

### 1. Calibre Installation

Calibre is required for MOBI to EPUB conversion. Install it in user space:

```bash
# Download and install Calibre binary
wget -nv -O- https://download.calibre-ebook.com/linux-installer.sh | sh /dev/stdin install_dir=~/calibre-bin

# Install missing libxcb-cursor dependency
mkdir -p ~/.local/libxcb-cursor0/usr/lib/x86_64-linux-gnu
wget -O ~/.local/libxcb-cursor0/usr/lib/x86_64-linux-gnu/libxcb-cursor.so.0 \
  http://archive.ubuntu.com/ubuntu/pool/main/libx/libxcb/libxcb-cursor0_1.15-1_amd64.deb
```

### 2. Virtual Environment Setup

The virtual environment is automatically configured with Calibre support via `postactivate` scripts:

```bash
# Activate virtual environment (Calibre setup is automatic)
source .venv/bin/activate

# Verify Calibre is available
ebook-convert --version
```

### 3. Anna's Archive MCP Server

Install the anna-mcp server for search and download functionality:

```bash
# Install anna-mcp (if not already installed)
pip install anna-mcp

# Set environment variables
export ANNAS_SECRET_KEY="your_secret_key"
export ANNAS_DOWNLOAD_PATH="/path/to/download/directory"
```

## Usage

### Basic Download Workflow

```python
from download_manager import BookDownloadManager

# Initialize download manager
manager = BookDownloadManager(
    csv_path="data/processed/sample_books_for_download.csv",
    download_dir="organized_outputs/anna_archive_download",
    daily_limit=10  # Respect rate limits
)

# Run a batch of downloads
summary = manager.run_download_batch(max_books=5)
print(f"Downloaded: {summary['downloaded']}, Failed: {summary['failed']}")
```

### Individual Book Processing

```python
# Search for a book
search_result = manager.search_book("Emma", "Jane Austen")

# Download the book
if search_result:
    success = manager.download_book(search_result, work_id=12345)
    if success:
        print("Download successful!")
```

### EPUB Guard Standalone Usage

```python
from aa_epub_guard import download_from_metadata, ensure_valid_epub

# Download with automatic format validation
metadata = {
    "file_unified_data": {
        "title_best": "Emma",
        "author_best": "Jane Austen",
        "ipfs_infos": [{"ipfs_cid": "QmfApP4c1A9YtDJor1TivVTLYpJpWkJB8BU55sUVQrHTuM"}]
    }
}

final_path = download_from_metadata(metadata, Path("./downloads"))
print(f"Downloaded to: {final_path}")
```

### Repair Existing Files

```bash
# Fix mislabeled files in download directory
python repair_existing_files.py
```

## Configuration

### Environment Variables

The system uses these environment variables (automatically set in virtual environment):

```bash
# Calibre configuration
export PATH="$HOME/calibre-bin/calibre:$PATH"
export EBOOK_CONVERT_BIN="$HOME/calibre-bin/calibre/ebook-convert"

# Qt headless mode
export QT_QPA_PLATFORM=offscreen
export LD_LIBRARY_PATH="$HOME/.local/libxcb-cursor0/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

# Anna's Archive MCP
export ANNAS_SECRET_KEY="your_secret_key"
export ANNAS_DOWNLOAD_PATH="/path/to/downloads"
```

### Download Manager Configuration

```python
manager = BookDownloadManager(
    csv_path="path/to/books.csv",           # CSV with book metadata
    download_dir="path/to/downloads",       # Download destination
    progress_file="download_progress.json", # Progress tracking file
    daily_limit=50                          # Daily download limit
)
```

## File Formats

### Supported Input Formats

- **EPUB** - Validated and normalized
- **MOBI** - Automatically converted to EPUB
- **ZIP** - Checked for valid EPUB structure
- **HTML** - Detected as error pages

### Output Format

All books are normalized to **EPUB** format with proper filenames:
- `Title - Author.epub`
- Fallback to MD5 hash if title/author unavailable

## Error Handling

### IPFS Gateway Resilience

The system handles common IPFS gateway issues:

- **403 Forbidden**: Uses proper User-Agent headers and exponential backoff
- **429 Rate Limited**: Automatic retry with increasing delays
- **Timeout**: 30-second timeout with fallback to next gateway
- **Network Errors**: Graceful degradation with detailed error messages

### EPUB Validation

- **Mimetype Check**: Ensures EPUB follows OCF specification
- **ZIP Structure**: Validates proper EPUB container format
- **Format Detection**: Sniffs actual file format vs. extension
- **Conversion**: MOBI files automatically converted to EPUB

## Progress Tracking

The system maintains detailed progress tracking:

```json
{
  "last_row": 150,
  "total_processed": 150,
  "total_downloaded": 120,
  "total_failed": 30,
  "last_run_date": "2025-09-28",
  "daily_downloads": 10,
  "download_history": [...]
}
```

## Troubleshooting

### Common Issues

1. **Calibre Not Found**
   ```bash
   # Verify installation
   ~/calibre-bin/calibre/ebook-convert --version
   
   # Check environment variables
   echo $EBOOK_CONVERT_BIN
   ```

2. **403 IPFS Errors**
   - The hardened fetcher should handle these automatically
   - Check network connectivity
   - Try different IPFS gateways

3. **MOBI Conversion Fails**
   ```bash
   # Test Calibre conversion manually
   ebook-convert input.mobi output.epub
   ```

4. **Virtual Environment Issues**
   ```bash
   # Reactivate environment
   deactivate
   source .venv/bin/activate
   ```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## API Reference

### BookDownloadManager

```python
class BookDownloadManager:
    def __init__(self, csv_path, download_dir, progress_file, daily_limit)
    def search_book(self, title: str, author_name: str) -> Optional[Dict]
    def download_book(self, search_result: Dict, work_id: int) -> bool
    def run_download_batch(self, max_books: Optional[int]) -> Dict
```

### EPUB Guard Functions

```python
def fetch_via_ipfs_cids(cids: Iterable[str], dest: Path, filename: str) -> Path
def sniff_format(path: Path) -> str
def ensure_valid_epub(path: Path, convert_mobi: bool, calibre_bin: str) -> Path
def download_from_metadata(meta: dict, out_dir: Path) -> Path
```

## Contributing

When modifying the download system:

1. **Test format detection** on various file types
2. **Verify Calibre integration** works in headless mode
3. **Check IPFS resilience** with different gateway combinations
4. **Update progress tracking** for new features
5. **Maintain backward compatibility** with existing downloads

## License

This component is part of the Romance Novel NLP Research project. See the main project LICENSE for details.

## Changelog

### v1.2.0 (2025-09-28)
- ✅ Hardened IPFS fetcher with User-Agent headers and backoff
- ✅ Persistent Calibre environment setup in virtual environment
- ✅ Enhanced error handling for 403/429 responses
- ✅ Improved EPUB guard validation and conversion

### v1.1.0
- ✅ EPUB guard integration with format detection
- ✅ MOBI to EPUB conversion via Calibre
- ✅ Progress tracking and resume capability

### v1.0.0
- ✅ Initial release with basic download functionality
- ✅ Anna's Archive MCP integration
- ✅ CSV-based batch processing
