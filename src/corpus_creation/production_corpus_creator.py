"""
Production Corpus Creator for Romance Novels

This module provides a production-ready solution for creating a romance novel corpus
using multiple methods including the annas-mcp tool, anna-dl, and direct HTTP requests.

Features:
- Multiple fallback methods for searching and downloading
- Rate limiting and quota management
- Comprehensive error handling and logging
- Metadata preservation and tracking
- Batch processing with progress tracking
"""

import sys
import json
import logging
import time
import csv
import subprocess
import os
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import re
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


@dataclass
class BookMetadata:
    """Comprehensive book metadata"""
    work_id: str
    title: str
    author: str
    publication_year: str
    genres: str
    series_title: str
    ratings_count: int
    average_rating: float
    goodreads_metadata: Dict[str, Any]
    
    # Anna's Archive specific
    anna_archive_url: Optional[str] = None
    anna_archive_md5: Optional[str] = None
    file_type: Optional[str] = None
    file_size: Optional[str] = None
    download_method: Optional[str] = None
    
    # Download tracking
    download_success: bool = False
    downloaded_file: Optional[str] = None
    download_timestamp: Optional[str] = None
    error_message: Optional[str] = None


class AnnaArchiveMCPClient:
    """Client for Anna's Archive MCP tool with error handling"""
    
    def __init__(self, 
                 annas_mcp_path: str,
                 api_key: str,
                 download_path: str,
                 logger: logging.Logger):
        self.annas_mcp_path = Path(annas_mcp_path)
        self.api_key = api_key
        self.download_path = Path(download_path)
        self.logger = logger
        
        # Download tracking
        self.downloads_today = 0
        self.download_limit = 25  # Brilliant Bookworm limit
        self.last_reset = datetime.now().date()
        
        # Validate setup
        self._validate_setup()
    
    def _validate_setup(self):
        """Validate annas-mcp setup"""
        if not self.annas_mcp_path.exists():
            raise FileNotFoundError(f"annas-mcp not found: {self.annas_mcp_path}")
        
        self.download_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"AnnaArchiveMCPClient initialized: {self.download_path}")
    
    def _reset_daily_counter(self):
        """Reset daily download counter if new day"""
        today = datetime.now().date()
        if today > self.last_reset:
            self.downloads_today = 0
            self.last_reset = today
            self.logger.info("Daily download counter reset")
    
    def search_book(self, author: str, title: str) -> List[Dict[str, Any]]:
        """Search for a book using annas-mcp"""
        self._reset_daily_counter()
        
        query = f"{author} {title}"
        self.logger.info(f"Searching with annas-mcp: {query}")
        
        try:
            env = os.environ.copy()
            env['ANNAS_SECRET_KEY'] = self.api_key
            env['ANNAS_DOWNLOAD_PATH'] = str(self.download_path)
            
            cmd = [str(self.annas_mcp_path), "search", query]
            result = subprocess.run(
                cmd, env=env, capture_output=True, text=True, timeout=60
            )
            
            if result.returncode == 0:
                # Parse the output (this will need to be adapted based on actual format)
                results = self._parse_mcp_output(result.stdout)
                self.logger.info(f"Found {len(results)} results")
                return results
            else:
                self.logger.error(f"annas-mcp search failed: {result.stderr}")
                return []
                
        except subprocess.TimeoutExpired:
            self.logger.error("annas-mcp search timeout")
            return []
        except Exception as e:
            self.logger.error(f"annas-mcp search error: {str(e)}")
            return []
    
    def _parse_mcp_output(self, output: str) -> List[Dict[str, Any]]:
        """Parse annas-mcp output (placeholder - needs actual format)"""
        # This is a placeholder - we need to see the actual output format
        # For now, return empty list to indicate no results
        self.logger.debug(f"Raw annas-mcp output: {output}")
        return []
    
    def download_book(self, md5: str, filename: str) -> bool:
        """Download a book using annas-mcp"""
        self._reset_daily_counter()
        
        if self.downloads_today >= self.download_limit:
            self.logger.warning(f"Daily download limit reached: {self.downloads_today}/{self.download_limit}")
            return False
        
        self.logger.info(f"Downloading with annas-mcp: {md5}")
        
        try:
            env = os.environ.copy()
            env['ANNAS_SECRET_KEY'] = self.api_key
            env['ANNAS_DOWNLOAD_PATH'] = str(self.download_path)
            
            cmd = [str(self.annas_mcp_path), "download", md5, filename]
            result = subprocess.run(
                cmd, env=env, capture_output=True, text=True, timeout=300
            )
            
            if result.returncode == 0:
                self.downloads_today += 1
                self.logger.info(f"Download successful: {filename}")
                return True
            else:
                self.logger.error(f"Download failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Download error: {str(e)}")
            return False
    
    def get_quota_status(self) -> Dict[str, Any]:
        """Get current quota status"""
        self._reset_daily_counter()
        return {
            "downloads_today": self.downloads_today,
            "download_limit": self.download_limit,
            "remaining": self.download_limit - self.downloads_today,
            "reset_date": str(self.last_reset)
        }


class AnnaArchiveHTTPClient:
    """HTTP client for Anna's Archive as fallback"""
    
    def __init__(self, download_path: str, logger: logging.Logger):
        self.download_path = Path(download_path)
        self.logger = logger
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        })
    
    def search_book(self, author: str, title: str) -> List[Dict[str, Any]]:
        """Search using HTTP requests"""
        query = f"{author} {title}"
        self.logger.info(f"HTTP search: {query}")
        
        try:
            # Try the JSON endpoint
            url = "https://annas-archive.org/search"
            params = {'format': 'json', 'q': query}
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    results = []
                    for item in data.get('results', [])[:5]:
                        results.append({
                            'title': item.get('title', ''),
                            'author': item.get('author', ''),
                            'url': item.get('url', ''),
                            'file_type': item.get('file_type', 'epub'),
                            'file_size': item.get('file_size', ''),
                            'method': 'http_json'
                        })
                    return results
                except json.JSONDecodeError:
                    pass
            
            # Fallback to HTML parsing (simplified)
            return self._search_html(query)
            
        except Exception as e:
            self.logger.error(f"HTTP search failed: {str(e)}")
            return []
    
    def _search_html(self, query: str) -> List[Dict[str, Any]]:
        """Simplified HTML search (placeholder)"""
        # This would need proper HTML parsing implementation
        self.logger.info("HTML search not implemented yet")
        return []
    
    def download_book(self, url: str, filename: str) -> bool:
        """Download using HTTP"""
        try:
            full_url = f"https://annas-archive.org{url}" if url.startswith('/') else url
            
            response = self.session.get(full_url, stream=True, timeout=60)
            response.raise_for_status()
            
            file_path = self.download_path / filename
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            self.logger.info(f"HTTP download successful: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"HTTP download failed: {str(e)}")
            return False


class ProductionCorpusCreator:
    """Production corpus creator with multiple methods"""
    
    def __init__(self, 
                 download_path: str,
                 api_key: str,
                 annas_mcp_path: str,
                 logger: logging.Logger):
        self.download_path = Path(download_path)
        self.logger = logger
        
        # Initialize clients
        self.mcp_client = AnnaArchiveMCPClient(
            annas_mcp_path, api_key, str(download_path), logger
        )
        self.http_client = AnnaArchiveHTTPClient(str(download_path), logger)
        
        # Tracking
        self.processed_books = []
        self.successful_downloads = []
        self.failed_downloads = []
        self.quota_exceeded = False
    
    def load_books_from_csv(self, csv_path: str) -> List[BookMetadata]:
        """Load books from CSV with comprehensive metadata"""
        books = []
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('title') and row.get('author_name'):
                        book = BookMetadata(
                            work_id=row.get('work_id', ''),
                            title=row['title'],
                            author=row['author_name'],
                            publication_year=row.get('publication_year', ''),
                            genres=row.get('genres_str', ''),
                            series_title=row.get('series_title', ''),
                            ratings_count=int(row.get('ratings_count_sum', 0)),
                            average_rating=float(row.get('average_rating_weighted_mean', 0)),
                            goodreads_metadata=row
                        )
                        books.append(book)
                        
        except Exception as e:
            self.logger.error(f"Error loading CSV: {str(e)}")
            raise
            
        self.logger.info(f"Loaded {len(books)} books from CSV")
        return books
    
    def search_book(self, book: BookMetadata) -> Optional[Dict[str, Any]]:
        """Search for a book using multiple methods"""
        self.logger.info(f"Searching: {book.title} by {book.author}")
        
        # Try MCP client first
        try:
            results = self.mcp_client.search_book(book.author, book.title)
            if results:
                result = results[0]
                result['method'] = 'mcp'
                return result
        except Exception as e:
            self.logger.warning(f"MCP search failed: {str(e)}")
        
        # Fallback to HTTP client
        try:
            results = self.http_client.search_book(book.author, book.title)
            if results:
                return results[0]
        except Exception as e:
            self.logger.warning(f"HTTP search failed: {str(e)}")
        
        return None
    
    def download_book(self, book: BookMetadata, search_result: Dict[str, Any]) -> bool:
        """Download a book using the best available method"""
        method = search_result.get('method', 'http')
        filename = f"{book.work_id}_{book.title.replace(' ', '_')}.{search_result.get('file_type', 'epub')}"
        
        self.logger.info(f"Downloading {book.title} using {method}")
        
        try:
            if method == 'mcp' and search_result.get('md5'):
                success = self.mcp_client.download_book(
                    search_result['md5'], filename
                )
            else:
                success = self.http_client.download_book(
                    search_result['url'], filename
                )
            
            if success:
                book.download_success = True
                book.downloaded_file = str(self.download_path / filename)
                book.download_timestamp = datetime.now().isoformat()
                book.download_method = method
                book.anna_archive_url = search_result.get('url')
                book.file_type = search_result.get('file_type')
                book.file_size = search_result.get('file_size')
                
                self.successful_downloads.append(book)
                return True
            else:
                book.error_message = f"Download failed using {method}"
                self.failed_downloads.append(book)
                return False
                
        except Exception as e:
            book.error_message = str(e)
            self.failed_downloads.append(book)
            self.logger.error(f"Download error for {book.title}: {str(e)}")
            return False
    
    def process_books(self, books: List[BookMetadata], max_books: Optional[int] = None) -> Dict[str, Any]:
        """Process books with comprehensive tracking"""
        if max_books:
            books = books[:max_books]
            self.logger.info(f"Processing first {max_books} books")
        
        results = {
            'total_books': len(books),
            'successful_downloads': 0,
            'failed_searches': 0,
            'failed_downloads': 0,
            'quota_exceeded': False,
            'processed_books': []
        }
        
        for i, book in enumerate(books, 1):
            self.logger.info(f"Processing {i}/{len(books)}: {book.title}")
            
            try:
                # Check quota
                quota_status = self.mcp_client.get_quota_status()
                if quota_status['remaining'] <= 0:
                    self.logger.warning("Daily quota exceeded, stopping downloads")
                    results['quota_exceeded'] = True
                    break
                
                # Search for the book
                search_result = self.search_book(book)
                
                if not search_result:
                    book.error_message = "No search results found"
                    results['failed_searches'] += 1
                    self.failed_downloads.append(book)
                    continue
                
                # Download the book
                download_success = self.download_book(book, search_result)
                
                if download_success:
                    results['successful_downloads'] += 1
                else:
                    results['failed_downloads'] += 1
                
                # Add to processed books
                results['processed_books'].append(asdict(book))
                
                # Rate limiting
                time.sleep(2)
                
            except Exception as e:
                book.error_message = str(e)
                results['failed_downloads'] += 1
                self.failed_downloads.append(book)
                self.logger.error(f"Error processing {book.title}: {str(e)}")
        
        self.logger.info(f"Processing complete: {results['successful_downloads']} successful, "
                        f"{results['failed_searches']} failed searches, "
                        f"{results['failed_downloads']} failed downloads")
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save comprehensive results"""
        results['timestamp'] = datetime.now().isoformat()
        results['download_path'] = str(self.download_path)
        results['quota_status'] = self.mcp_client.get_quota_status()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results saved to: {output_path}")


def setup_production_logging(log_dir: str) -> logging.Logger:
    """Set up production logging"""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"production_corpus_creation_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("production_corpus_creation")
    logger.info(f"Production logging initialized: {log_file}")
    return logger


def main():
    """Main function for production corpus creation"""
    print("Production Corpus Creator for Romance Novels")
    print("="*60)
    
    # Setup paths
    download_path = project_root / "data" / "raw" / "anna_archive_corpus"
    sample_csv_path = project_root / "data" / "processed" / "sample_books_for_download.csv"
    output_path = project_root / "data" / "intermediate" / "anna_archive_metadata" / "production_corpus_results.json"
    annas_mcp_path = project_root / "annas-mcp"
    
    # Setup logging
    log_dir = project_root / "logs" / "corpus_creation"
    logger = setup_production_logging(str(log_dir))
    
    # Load API key
    config_path = project_root / "anna_archive_config.env"
    api_key = None
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            for line in f:
                if line.startswith('ANNAS_SECRET_KEY='):
                    api_key = line.split('=', 1)[1].strip()
                    break
    
    if not api_key or api_key == 'your_secret_key_here':
        logger.error("No valid API key found in configuration")
        return False
    
    # Create corpus creator
    creator = ProductionCorpusCreator(
        str(download_path), api_key, str(annas_mcp_path), logger
    )
    
    try:
        # Load and process books
        books = creator.load_books_from_csv(str(sample_csv_path))
        
        # Process first 2 books for testing
        results = creator.process_books(books, max_books=2)
        
        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        creator.save_results(results, str(output_path))
        
        # Print summary
        print(f"\nProduction Results Summary:")
        print(f"  Total books processed: {results['total_books']}")
        print(f"  Successful downloads: {results['successful_downloads']}")
        print(f"  Failed searches: {results['failed_searches']}")
        print(f"  Failed downloads: {results['failed_downloads']}")
        print(f"  Quota exceeded: {results['quota_exceeded']}")
        print(f"  Results saved to: {output_path}")
        
        if results['quota_status']:
            quota = results['quota_status']
            print(f"  Quota status: {quota['downloads_today']}/{quota['download_limit']} used")
        
        return True
        
    except Exception as e:
        logger.error(f"Production corpus creation failed: {str(e)}")
        print(f"Error: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
