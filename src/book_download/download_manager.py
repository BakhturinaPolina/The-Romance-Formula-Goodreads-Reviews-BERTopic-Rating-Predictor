#!/usr/bin/env python3
"""
Book Download Research Component - Download Manager
Main system for downloading romance novels from Anna's Archive with progress tracking
"""

import pandas as pd
import os
import sys
import logging
import time
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Import MCP integration
from mcp_integration import AnnaMCPIntegration

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BookDownloadManager:
    """Manages downloading books from Anna's Archive with progress tracking"""
    
    def __init__(self, 
                 csv_path: str,
                 download_dir: str = "/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/organized_outputs/anna_archive_download",
                 progress_file: str = "download_progress.json",
                 daily_limit: int = 999):
        """
        Initialize the download manager
        
        Args:
            csv_path: Path to the CSV file with book metadata
            download_dir: Directory to save downloaded books
            progress_file: File to track download progress
            daily_limit: Maximum books to download per day
        """
        self.csv_path = csv_path
        self.download_dir = download_dir
        self.progress_file = progress_file
        self.daily_limit = daily_limit
        
        # Create download directory if it doesn't exist
        os.makedirs(self.download_dir, exist_ok=True)
        
        # Initialize MCP integration
        self.mcp = AnnaMCPIntegration()
        
        # Load progress tracking
        self.progress = self._load_progress()
        
        logger.info(f"Download Manager initialized:")
        logger.info(f"  CSV: {csv_path}")
        logger.info(f"  Download dir: {download_dir}")
        logger.info(f"  Progress file: {progress_file}")
        logger.info(f"  Daily limit: {daily_limit}")
    
    def _load_progress(self) -> Dict:
        """Load progress from file or create new progress tracking"""
        progress_path = os.path.join(self.download_dir, self.progress_file)
        
        if os.path.exists(progress_path):
            try:
                with open(progress_path, 'r') as f:
                    progress = json.load(f)
                logger.info(f"Loaded existing progress: {progress.get('last_row', 0)} books processed")
                return progress
            except Exception as e:
                logger.warning(f"Error loading progress file: {e}. Creating new progress tracking.")
        
        # Create new progress tracking
        progress = {
            'last_row': 0,
            'total_processed': 0,
            'total_downloaded': 0,
            'total_failed': 0,
            'last_run_date': None,
            'daily_downloads': 0,
            'download_history': []
        }
        logger.info("Created new progress tracking")
        return progress
    
    def _save_progress(self):
        """Save current progress to file"""
        progress_path = os.path.join(self.download_dir, self.progress_file)
        try:
            # Convert numpy types to Python native types for JSON serialization
            progress_copy = self._convert_numpy_types(self.progress)
            with open(progress_path, 'w') as f:
                json.dump(progress_copy, f, indent=2)
            logger.debug(f"Progress saved to {progress_path}")
        except Exception as e:
            logger.error(f"Error saving progress: {e}")
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    def _reset_daily_counter_if_needed(self):
        """Reset daily download counter if it's a new day"""
        today = datetime.now().date().isoformat()
        if self.progress.get('last_run_date') != today:
            logger.info(f"New day detected ({today}). Resetting daily counter.")
            self.progress['daily_downloads'] = 0
            self.progress['last_run_date'] = today
            self._save_progress()
    
    def _can_download_more_today(self) -> bool:
        """Check if we can download more books today"""
        return self.progress['daily_downloads'] < self.daily_limit
    
    def search_book(self, title: str, author_name: str) -> Optional[Dict]:
        """
        Search for a book using anna-mcp server
        """
        logger.info(f"Searching for: '{title}' by {author_name}")
        
        # Create search term combining title and author
        search_term = f"{title} {author_name}"
        logger.info(f"Search term: '{search_term}'")
        
        # Use real MCP integration
        search_results = self.mcp.search_books(search_term)
        
        if not search_results:
            logger.warning(f"No search results found for: {search_term}")
            return None
        
        # Return the first result
        first_result = search_results[0]
        
        # If title/author are empty, use the original search terms
        if not first_result.get('title') or first_result.get('title') == 'Unknown Title':
            first_result['title'] = title
        if not first_result.get('author') or first_result.get('author') == 'Unknown Author':
            first_result['author'] = author_name
        
        logger.info(f"Found book: {first_result.get('title', 'Unknown')} by {first_result.get('author', 'Unknown')}")
        
        return first_result
    
    def download_book(self, search_result: Dict, work_id: int) -> bool:
        """
        Download a book using Anna's Archive API or anna-mcp server
        """
        logger.info(f"Downloading book: {search_result.get('title', 'Unknown')}")
        
        # Use real MCP integration
        book_hash = search_result.get('hash')
        title = search_result.get('title', f'book_{work_id}')
        format_type = search_result.get('format', 'epub')
        
        if not book_hash:
            logger.error("No hash found in search result")
            return False
        
        # Use MCP integration to download
        success = self.mcp.download_book(book_hash, title, format_type)
        
        if success:
            logger.info(f"Successfully downloaded: {title}")
        else:
            logger.error(f"Failed to download: {title}")
        
        return success
    
    def download_by_md5(self, md5_or_url: str, work_id: int, title: str = None) -> Dict:
        """
        Download a book directly by MD5 hash or Anna's Archive URL
        
        Args:
            md5_or_url: MD5 hash or Anna's Archive URL
            work_id: Work ID for tracking
            title: Optional title for filename
            
        Returns:
            Dictionary with download result
        """
        logger.info(f"Downloading book by MD5/URL: {md5_or_url}")
        
        # Create filename if title provided
        filename = None
        if title:
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"{work_id}_{safe_title}"
        
        # Use MCP integration for direct download
        result = self.mcp.download_by_md5(md5_or_url, filename)
        
        if result.get("success"):
            logger.info(f"Successfully downloaded: {result['filepath']}")
        else:
            logger.error(f"Failed to download: {result.get('message', 'Unknown error')}")
        
        return result
    
    def process_single_book_by_md5(self, row: pd.Series) -> Dict:
        """Process a single book using MD5 hash (skip search)"""
        work_id = row['work_id']
        title = row['title']
        author_name = row['author_name']
        md5_hash = row.get('md5_hash') or row.get('md5') or row.get('hash')
        
        logger.info(f"Processing book {work_id} by MD5: '{title}' by {author_name}")
        
        result = {
            'work_id': work_id,
            'title': title,
            'author_name': author_name,
            'publication_year': row.get('publication_year'),
            'status': 'failed',
            'error': None,
            'download_path': None,
            'timestamp': datetime.now().isoformat(),
            'method': 'md5_direct'
        }
        
        if not md5_hash:
            result['error'] = 'No MD5 hash found in row data'
            logger.warning(f"No MD5 hash found for book: {title}")
            return result
        
        try:
            # Download directly by MD5
            download_result = self.download_by_md5(md5_hash, work_id, title)
            
            if download_result.get('success'):
                result['status'] = 'downloaded'
                result['download_path'] = download_result['filepath']
                result['file_size'] = download_result.get('file_size')
                logger.info(f"Successfully downloaded: {title}")
            else:
                result['error'] = download_result.get('message', 'Download failed')
                logger.error(f"Download failed: {title} - {result['error']}")
        
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error processing book {work_id}: {e}")
        
        return result
    
    def process_single_book(self, row: pd.Series) -> Dict:
        """Process a single book (search + download)"""
        work_id = row['work_id']
        title = row['title']
        author_name = row['author_name']
        
        logger.info(f"Processing book {work_id}: '{title}' by {author_name}")
        
        result = {
            'work_id': work_id,
            'title': title,
            'author_name': author_name,
            'publication_year': row['publication_year'],
            'status': 'failed',
            'error': None,
            'download_path': None,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Search for the book
            search_result = self.search_book(title, author_name)
            if not search_result or not search_result.get('hash'):
                result['error'] = 'Book not found in Anna\'s Archive'
                logger.warning(f"Book not found: {title}")
                return result
            
            # Download the book
            if self.download_book(search_result, work_id):
                result['status'] = 'downloaded'
                result['download_path'] = os.path.join(
                    self.download_dir, 
                    f"{work_id}_{title.replace(' ', '_')}.{search_result['format']}"
                )
                logger.info(f"Successfully downloaded: {title}")
            else:
                result['error'] = 'Download failed'
                logger.error(f"Download failed: {title}")
        
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error processing book {work_id}: {e}")
        
        return result
    
    def run_download_batch(self, max_books: Optional[int] = None) -> Dict:
        """
        Run a batch of downloads, respecting daily limits
        
        Args:
            max_books: Maximum number of books to process in this run
            
        Returns:
            Summary of the download batch
        """
        logger.info("=== STARTING DOWNLOAD BATCH ===")
        
        # Reset daily counter if needed
        self._reset_daily_counter_if_needed()
        
        # Check if we can download more today
        if not self._can_download_more_today():
            logger.warning(f"Daily limit reached ({self.daily_limit}). Skipping download batch.")
            return {'status': 'daily_limit_reached', 'processed': 0}
        
        # Load CSV data
        try:
            df = pd.read_csv(self.csv_path)
            logger.info(f"Loaded {len(df)} books from CSV")
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return {'status': 'error', 'error': str(e)}
        
        # Determine how many books to process
        remaining_today = self.daily_limit - self.progress['daily_downloads']
        start_row = self.progress['last_row']
        remaining_total = len(df) - start_row
        
        if max_books:
            books_to_process = min(max_books, remaining_today, remaining_total)
        else:
            books_to_process = min(remaining_today, remaining_total)
        
        logger.info(f"Processing {books_to_process} books (starting from row {start_row})")
        
        if books_to_process <= 0:
            logger.info("No books to process")
            return {'status': 'no_books_to_process', 'processed': 0}
        
        # Process books
        batch_results = []
        processed_count = 0
        
        for i in range(start_row, start_row + books_to_process):
            if i >= len(df):
                break
            
            row = df.iloc[i]
            logger.info(f"\n--- Processing book {i+1}/{len(df)} ---")
            
            # Process the book
            result = self.process_single_book(row)
            batch_results.append(result)
            
            # Update progress
            self.progress['last_row'] = i + 1
            self.progress['total_processed'] += 1
            
            if result['status'] == 'downloaded':
                self.progress['total_downloaded'] += 1
                self.progress['daily_downloads'] += 1
            else:
                self.progress['total_failed'] += 1
            
            # Add to history
            self.progress['download_history'].append(result)
            
            # Save progress after each book
            self._save_progress()
            
            processed_count += 1
            
            # Check if we've reached daily limit
            if not self._can_download_more_today():
                logger.info(f"Daily limit reached. Stopping at book {i+1}")
                break
            
            # Add delay between downloads
            time.sleep(2)
        
        # Save final progress
        self._save_progress()
        
        # Create summary
        summary = {
            'status': 'completed',
            'processed': processed_count,
            'downloaded': sum(1 for r in batch_results if r['status'] == 'downloaded'),
            'failed': sum(1 for r in batch_results if r['status'] == 'failed'),
            'last_row': self.progress['last_row'],
            'daily_downloads': self.progress['daily_downloads'],
            'remaining_today': self.daily_limit - self.progress['daily_downloads'],
            'results': batch_results
        }
        
        logger.info("=== DOWNLOAD BATCH COMPLETED ===")
        logger.info(f"Processed: {summary['processed']}")
        logger.info(f"Downloaded: {summary['downloaded']}")
        logger.info(f"Failed: {summary['failed']}")
        logger.info(f"Last row: {summary['last_row']}")
        logger.info(f"Daily downloads: {summary['daily_downloads']}/{self.daily_limit}")

        # Ensure summary is JSON serializable
        return self._convert_numpy_types(summary)
    
    def run_md5_download_batch(self, max_books: Optional[int] = None) -> Dict:
        """
        Run a batch of MD5-based downloads, respecting daily limits
        
        Args:
            max_books: Maximum number of books to process in this run
            
        Returns:
            Summary of the download batch
        """
        logger.info("=== STARTING MD5 DOWNLOAD BATCH ===")
        
        # Reset daily counter if needed
        self._reset_daily_counter_if_needed()
        
        # Check if we can download more today
        if not self._can_download_more_today():
            logger.warning(f"Daily limit reached ({self.daily_limit}). Skipping download batch.")
            return {'status': 'daily_limit_reached', 'processed': 0}
        
        # Load CSV data
        try:
            df = pd.read_csv(self.csv_path)
            logger.info(f"Loaded {len(df)} books from CSV")
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return {'status': 'error', 'error': str(e)}
        
        # Filter rows that have MD5 hashes
        md5_conditions = []
        for col in ['md5_hash', 'md5', 'hash']:
            if col in df.columns:
                md5_conditions.append(df[col].notna())
        
        if md5_conditions:
            # Combine conditions with OR logic
            combined_condition = md5_conditions[0]
            for condition in md5_conditions[1:]:
                combined_condition = combined_condition | condition
            md5_rows = df[combined_condition]
        else:
            md5_rows = df.iloc[0:0]  # Empty dataframe
        logger.info(f"Found {len(md5_rows)} books with MD5 hashes")
        
        if len(md5_rows) == 0:
            logger.warning("No books with MD5 hashes found in CSV")
            return {'status': 'no_md5_books', 'processed': 0}
        
        # Determine how many books to process
        remaining_today = self.daily_limit - self.progress['daily_downloads']
        start_row = self.progress['last_row']
        remaining_total = len(md5_rows) - start_row
        
        if max_books:
            books_to_process = min(max_books, remaining_today, remaining_total)
        else:
            books_to_process = min(remaining_today, remaining_total)
        
        logger.info(f"Processing {books_to_process} books with MD5 hashes (starting from row {start_row})")
        
        if books_to_process <= 0:
            logger.info("No books to process")
            return {'status': 'no_books_to_process', 'processed': 0}
        
        # Process books
        batch_results = []
        processed_count = 0
        
        for i in range(start_row, start_row + books_to_process):
            if i >= len(md5_rows):
                break
            
            row = md5_rows.iloc[i]
            logger.info(f"\n--- Processing book {i+1}/{len(md5_rows)} ---")
            
            # Process the book by MD5
            result = self.process_single_book_by_md5(row)
            batch_results.append(result)
            
            # Update progress
            self.progress['last_row'] = i + 1
            self.progress['total_processed'] += 1
            
            if result['status'] == 'downloaded':
                self.progress['total_downloaded'] += 1
                self.progress['daily_downloads'] += 1
            else:
                self.progress['total_failed'] += 1
            
            # Add to history
            self.progress['download_history'].append(result)
            
            # Save progress after each book
            self._save_progress()
            
            processed_count += 1
            
            # Check if we've reached daily limit
            if not self._can_download_more_today():
                logger.info(f"Daily limit reached. Stopping at book {i+1}")
                break
            
            # Add delay between downloads
            time.sleep(2)
        
        # Save final progress
        self._save_progress()
        
        # Create summary
        summary = {
            'status': 'completed',
            'processed': processed_count,
            'downloaded': sum(1 for r in batch_results if r['status'] == 'downloaded'),
            'failed': sum(1 for r in batch_results if r['status'] == 'failed'),
            'last_row': self.progress['last_row'],
            'daily_downloads': self.progress['daily_downloads'],
            'remaining_today': self.daily_limit - self.progress['daily_downloads'],
            'results': batch_results,
            'method': 'md5_direct'
        }
        
        logger.info("=== MD5 DOWNLOAD BATCH COMPLETED ===")
        logger.info(f"Processed: {summary['processed']}")
        logger.info(f"Downloaded: {summary['downloaded']}")
        logger.info(f"Failed: {summary['failed']}")
        logger.info(f"Last row: {summary['last_row']}")
        logger.info(f"Daily downloads: {summary['daily_downloads']}/{self.daily_limit}")

        # Ensure summary is JSON serializable
        return self._convert_numpy_types(summary)

def main():
    """Main function for testing the download manager"""
    
    # Test with sample CSV
    sample_csv = "/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/data/processed/sample_books_for_download.csv"
    
    logger.info("=== TESTING DOWNLOAD MANAGER ===")
    
    # Initialize download manager
    manager = BookDownloadManager(
        csv_path=sample_csv,
        daily_limit=5  # Small limit for testing
    )
    
    # Run a small batch
    summary = manager.run_download_batch(max_books=3)
    
    logger.info(f"Test completed: {summary}")

if __name__ == "__main__":
    main()
