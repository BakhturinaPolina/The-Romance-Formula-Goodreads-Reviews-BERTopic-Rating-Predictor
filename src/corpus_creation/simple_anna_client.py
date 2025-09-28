"""
Simple Anna's Archive Client for Romance Novel Corpus Creation

This module provides a simplified interface for downloading romance novels
from Anna's Archive using the existing anna-dl tool and direct HTTP requests.

Based on the existing corpus creation infrastructure in the project.
"""

import json
import logging
import os
import subprocess
import time
import csv
import requests
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import re


@dataclass
class BookResult:
    """Represents a book search result"""
    title: str
    author: str
    url: str
    file_type: str
    file_size: str
    language: str = "en"
    year: Optional[str] = None


class SimpleAnnaClient:
    """Simple client for Anna's Archive using direct HTTP requests"""
    
    def __init__(self, download_path: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the simple Anna's Archive client
        
        Args:
            download_path: Directory to download files to
            logger: Optional logger instance
        """
        self.download_path = Path(download_path)
        self.logger = logger or logging.getLogger(__name__)
        self.base_url = "https://annas-archive.org"
        self.search_url = f"{self.base_url}/search"
        
        # Create download directory
        self.download_path.mkdir(parents=True, exist_ok=True)
        
        # Session for requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        self.logger.info(f"SimpleAnnaClient initialized with download path: {self.download_path}")
    
    def search_books(self, query: str, max_results: int = 5) -> List[BookResult]:
        """
        Search for books using direct HTTP requests to Anna's Archive
        
        Args:
            query: Search query (author + title)
            max_results: Maximum number of results to return
            
        Returns:
            List of BookResult objects
        """
        self.logger.info(f"Searching for: {query}")
        
        try:
            # Use the JSON endpoint for search
            params = {
                'format': 'json',
                'q': query
            }
            
            response = self.session.get(self.search_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse JSON response
            data = response.json()
            
            results = []
            for item in data.get('results', [])[:max_results]:
                try:
                    book_result = BookResult(
                        title=item.get('title', ''),
                        author=item.get('author', ''),
                        url=item.get('url', ''),
                        file_type=item.get('file_type', ''),
                        file_size=item.get('file_size', ''),
                        language=item.get('language', 'en'),
                        year=item.get('year')
                    )
                    results.append(book_result)
                except Exception as e:
                    self.logger.warning(f"Error parsing search result: {str(e)}")
                    continue
            
            self.logger.info(f"Found {len(results)} results for query: {query}")
            return results
            
        except requests.RequestException as e:
            self.logger.error(f"Search request failed for '{query}': {str(e)}")
            return []
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse search response for '{query}': {str(e)}")
            return []
        except Exception as e:
            self.logger.error(f"Search error for '{query}': {str(e)}")
            return []
    
    def get_download_url(self, book_url: str) -> Optional[str]:
        """
        Get the direct download URL for a book
        
        Args:
            book_url: URL to the book page on Anna's Archive
            
        Returns:
            Direct download URL or None if not found
        """
        try:
            full_url = f"{self.base_url}{book_url}" if book_url.startswith('/') else book_url
            response = self.session.get(full_url, timeout=30)
            response.raise_for_status()
            
            # Look for download links in the HTML
            html = response.text
            
            # Try to find direct download links
            download_patterns = [
                r'href="([^"]*\.(?:epub|pdf|mobi|azw3)[^"]*)"',
                r'href="([^"]*download[^"]*)"',
                r'data-download-url="([^"]*)"'
            ]
            
            for pattern in download_patterns:
                matches = re.findall(pattern, html, re.IGNORECASE)
                if matches:
                    download_url = matches[0]
                    if download_url.startswith('/'):
                        download_url = f"{self.base_url}{download_url}"
                    return download_url
            
            self.logger.warning(f"No download URL found for: {book_url}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting download URL for '{book_url}': {str(e)}")
            return None
    
    def download_book(self, book_result: BookResult, filename: Optional[str] = None) -> Optional[Path]:
        """
        Download a book from Anna's Archive
        
        Args:
            book_result: BookResult object with book information
            filename: Optional custom filename
            
        Returns:
            Path to downloaded file or None if failed
        """
        try:
            # Get download URL
            download_url = self.get_download_url(book_result.url)
            if not download_url:
                return None
            
            # Generate filename
            if not filename:
                safe_title = re.sub(r'[^\w\s-]', '', book_result.title)
                safe_author = re.sub(r'[^\w\s-]', '', book_result.author)
                filename = f"{safe_author}_{safe_title}.{book_result.file_type}"
                filename = re.sub(r'\s+', '_', filename)
            
            file_path = self.download_path / filename
            
            # Download the file
            self.logger.info(f"Downloading: {book_result.title} to {file_path}")
            
            response = self.session.get(download_url, stream=True, timeout=60)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            self.logger.info(f"Downloaded: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Download failed for '{book_result.title}': {str(e)}")
            return None


class CorpusCreator:
    """Main class for creating romance novel corpus"""
    
    def __init__(self, 
                 client: SimpleAnnaClient,
                 metadata_csv_path: str,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the corpus creator
        
        Args:
            client: Configured SimpleAnnaClient
            metadata_csv_path: Path to CSV with book metadata
            logger: Optional logger instance
        """
        self.client = client
        self.metadata_csv_path = Path(metadata_csv_path)
        self.logger = logger or logging.getLogger(__name__)
        
        # Tracking
        self.processed_books = []
        self.failed_searches = []
        self.failed_downloads = []
        self.downloaded_files = []
    
    def load_book_metadata(self) -> List[Dict[str, Any]]:
        """Load book metadata from CSV file"""
        books = []
        
        try:
            with open(self.metadata_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('title') and row.get('author_name'):
                        books.append(row)
                        
        except Exception as e:
            self.logger.error(f"Error loading metadata CSV: {str(e)}")
            raise
            
        self.logger.info(f"Loaded {len(books)} books from metadata CSV")
        return books
    
    def process_books(self, books: List[Dict[str, Any]], max_books: Optional[int] = None) -> Dict[str, Any]:
        """
        Process books from metadata list
        
        Args:
            books: List of book metadata dictionaries
            max_books: Maximum number of books to process (for testing)
            
        Returns:
            Processing results summary
        """
        if max_books:
            books = books[:max_books]
            self.logger.info(f"Processing first {max_books} books for testing")
        
        results = {
            'total_books': len(books),
            'successful_downloads': 0,
            'failed_searches': 0,
            'failed_downloads': 0,
            'downloaded_files': []
        }
        
        for i, book in enumerate(books, 1):
            self.logger.info(f"Processing book {i}/{len(books)}: {book['title']} by {book['author_name']}")
            
            try:
                # Search for the book
                query = f"{book['author_name']} {book['title']}"
                search_results = self.client.search_books(query, max_results=3)
                
                if not search_results:
                    self.failed_searches.append(book)
                    results['failed_searches'] += 1
                    self.logger.warning(f"No search results for: {book['title']}")
                    continue
                
                # Try to download the first result
                best_match = search_results[0]
                downloaded_file = self.client.download_book(best_match)
                
                if downloaded_file:
                    results['successful_downloads'] += 1
                    results['downloaded_files'].append({
                        'goodreads_metadata': book,
                        'anna_archive_result': {
                            'title': best_match.title,
                            'author': best_match.author,
                            'file_type': best_match.file_type,
                            'file_size': best_match.file_size
                        },
                        'downloaded_file': str(downloaded_file)
                    })
                    self.downloaded_files.append(downloaded_file)
                else:
                    results['failed_downloads'] += 1
                    self.failed_downloads.append(book)
                
                # Small delay to be respectful
                time.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Error processing {book['title']}: {str(e)}")
                results['failed_downloads'] += 1
                self.failed_downloads.append(book)
        
        self.logger.info(f"Processing complete: {results['successful_downloads']} successful, "
                        f"{results['failed_searches']} failed searches, "
                        f"{results['failed_downloads']} failed downloads")
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save processing results to JSON file"""
        results['timestamp'] = time.time()
        results['metadata_csv'] = str(self.metadata_csv_path)
        results['download_path'] = str(self.client.download_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results saved to: {output_path}")


def setup_logging(log_dir: str, level: str = "INFO") -> logging.Logger:
    """Set up logging for corpus creation"""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped log file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"simple_corpus_creation_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("simple_corpus_creation")
    logger.info(f"Logging initialized: {log_file}")
    return logger


if __name__ == "__main__":
    # Example usage
    print("Simple Anna's Archive Client for Romance Novel Corpus Creation")
    print("This module should be imported and used with proper configuration.")
