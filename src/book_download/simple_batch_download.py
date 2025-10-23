#!/usr/bin/env python3
"""
Simple batch download script using the working AnnaAPIClient
"""

import sys
import os
import pandas as pd
import logging
from pathlib import Path

# Add current directory to path
sys.path.append('.')
from anna_api_client import AnnaAPIClient

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_books_from_csv(csv_path: str, output_dir: str, max_books: int = 10):
    """
    Download books from CSV with MD5 hashes
    
    Args:
        csv_path: Path to CSV file with MD5 hashes
        output_dir: Directory to save downloaded books
        max_books: Maximum number of books to download
    """
    
    # Initialize API client
    client = AnnaAPIClient()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read CSV
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} books from {csv_path}")
    
    # Find MD5 column
    md5_columns = ['md5_hash', 'md5', 'hash']
    md5_col = None
    for col in md5_columns:
        if col in df.columns:
            md5_col = col
            break
    
    if not md5_col:
        logger.error(f"No MD5 column found. Available columns: {list(df.columns)}")
        return
    
    logger.info(f"Using MD5 column: {md5_col}")
    
    # Filter rows with valid MD5 hashes
    # Convert to string and filter for 32-character MD5 hashes
    df[md5_col] = df[md5_col].astype(str)
    valid_md5s = df[df[md5_col].notna() & (df[md5_col] != 'nan') & (df[md5_col].str.len() == 32)]
    logger.info(f"Found {len(valid_md5s)} books with valid MD5 hashes")
    
    if len(valid_md5s) == 0:
        logger.error("No valid MD5 hashes found in CSV")
        return
    
    # Process books
    downloaded = 0
    failed = 0
    
    for idx, row in valid_md5s.head(max_books).iterrows():
        md5 = row[md5_col]
        title = row.get('title', f'book_{idx}')
        author = row.get('author_name', 'unknown')
        
        logger.info(f"[{downloaded + failed + 1}/{min(max_books, len(valid_md5s))}] Downloading: {title} by {author}")
        
        # Create safe filename
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{idx}_{safe_title}_{md5}.epub"
        
        try:
            result = client.download_book(md5, filename, output_dir)
            
            if result.get('success'):
                downloaded += 1
                logger.info(f"✅ Downloaded: {filename} ({result.get('file_size', 0)} bytes)")
            else:
                failed += 1
                logger.error(f"❌ Failed: {result.get('message', 'Unknown error')}")
                
        except Exception as e:
            failed += 1
            logger.error(f"❌ Error: {e}")
    
    logger.info(f"=== BATCH DOWNLOAD COMPLETE ===")
    logger.info(f"Downloaded: {downloaded}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total processed: {downloaded + failed}")

if __name__ == "__main__":
    # Default paths
    csv_path = "test_md5s.csv"
    output_dir = "../../organized_outputs/epub_downloads"
    max_books = 4  # Start with 4 books for testing
    
    logger.info("Simple Batch Download Script")
    logger.info("=" * 40)
    logger.info(f"CSV: {csv_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Max books: {max_books}")
    logger.info("")
    
    download_books_from_csv(csv_path, output_dir, max_books)
