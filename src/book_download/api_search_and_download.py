#!/usr/bin/env python3
"""
API-based Search and Download Script for Anna's Archive
Searches for books by title/author using the API, then downloads them
"""

import sys
import os
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Add current directory to path
sys.path.append('.')
from anna_api_client import AnnaAPIClient

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def search_and_download_books(csv_path: str, output_dir: str, max_books: int = 10):
    """
    Search for books by title/author and download them
    
    Args:
        csv_path: Path to CSV file with title, author_name columns
        output_dir: Directory to save downloaded books
        max_books: Maximum number of books to process
    """
    
    # Initialize API client
    client = AnnaAPIClient()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read CSV
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} books from {csv_path}")
    
    # Validate required columns
    required_cols = ['title', 'author_name']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        logger.error(f"Available columns: {list(df.columns)}")
        return
    
    # Process books
    results = []
    downloaded = 0
    failed = 0
    not_found = 0
    
    for idx, row in df.head(max_books).iterrows():
        title = str(row['title']).strip()
        author = str(row['author_name']).strip() if pd.notna(row['author_name']) else None
        
        logger.info(f"[{idx+1}/{min(max_books, len(df))}] Searching: '{title}' by {author}")
        
        # Search for the book using API
        try:
            md5 = client.search_md5(title, author)
            
            if md5:
                logger.info(f"✅ Found MD5: {md5}")
                
                # Create safe filename
                safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                filename = f"{idx}_{safe_title}_{md5}.epub"
                
                # Download the book
                result = client.download_book(md5, filename, output_dir)
                
                if result.get('success'):
                    downloaded += 1
                    logger.info(f"✅ Downloaded: {filename} ({result.get('file_size', 0)} bytes)")
                    results.append({
                        'work_id': row.get('work_id', idx),
                        'title': title,
                        'author': author,
                        'md5': md5,
                        'status': 'downloaded',
                        'filepath': result.get('filepath'),
                        'file_size': result.get('file_size')
                    })
                else:
                    failed += 1
                    logger.error(f"❌ Download failed: {result.get('message', 'Unknown error')}")
                    results.append({
                        'work_id': row.get('work_id', idx),
                        'title': title,
                        'author': author,
                        'md5': md5,
                        'status': 'download_failed',
                        'error': result.get('message', 'Unknown error')
                    })
            else:
                not_found += 1
                logger.warning(f"❌ Not found: '{title}' by {author}")
                results.append({
                    'work_id': row.get('work_id', idx),
                    'title': title,
                    'author': author,
                    'md5': None,
                    'status': 'not_found',
                    'error': 'Book not found in Anna\'s Archive'
                })
                
        except Exception as e:
            failed += 1
            logger.error(f"❌ Error processing '{title}': {e}")
            results.append({
                'work_id': row.get('work_id', idx),
                'title': title,
                'author': author,
                'md5': None,
                'status': 'error',
                'error': str(e)
            })
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(output_dir, 'search_and_download_results.csv')
    results_df.to_csv(results_csv, index=False)
    
    logger.info(f"=== SEARCH AND DOWNLOAD COMPLETE ===")
    logger.info(f"Total processed: {len(results)}")
    logger.info(f"Downloaded: {downloaded}")
    logger.info(f"Not found: {not_found}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Results saved to: {results_csv}")

if __name__ == "__main__":
    # Default paths - use the sample books CSV
    csv_path = "../../data/processed/sample_50_books.csv"
    output_dir = "../../organized_outputs/epub_downloads"
    max_books = 10  # Start with 10 books for testing
    
    logger.info("API Search and Download Script")
    logger.info("=" * 40)
    logger.info(f"CSV: {csv_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Max books: {max_books}")
    logger.info("")
    
    search_and_download_books(csv_path, output_dir, max_books)
