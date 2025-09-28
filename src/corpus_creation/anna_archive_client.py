"""
Anna's Archive Client for Romance Novel Corpus Creation

This module provides a Python interface to the annas-mcp tool for searching
and downloading romance novels from Anna's Archive with proper API key authentication.

Usage:
    client = AnnaArchiveClient(api_key="your_key", download_path="/path/to/downloads")
    results = client.search_book("Patricia Cabot", "A Little Scandal") 
    download_info = client.download_book(md5_hash="...", filename="...")
"""

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import csv


@dataclass
class BookSearchResult:
    """Represents a book search result from Anna's Archive"""
    title: str
    author: str
    md5: str
    extension: str
    filesize: str
    language: str
    year: Optional[str] = None
    publisher: Optional[str] = None
    additional_info: Dict[str, Any] = None

    def __post_init__(self):
        if self.additional_info is None:
            self.additional_info = {}


@dataclass 
class DownloadResult:
    """Represents a download operation result"""
    success: bool
    md5: str
    filepath: Optional[Path] = None
    error_message: Optional[str] = None
    download_time: Optional[float] = None


class AnnaArchiveClient:
    """Client for interacting with Anna's Archive via annas-mcp tool"""
    
    def __init__(self, 
                 annas_mcp_path: str,
                 api_key: str, 
                 download_path: str,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the Anna's Archive client
        
        Args:
            annas_mcp_path: Path to the annas-mcp binary
            api_key: Your Anna's Archive API secret key
            download_path: Directory to download files to
            logger: Optional logger instance
        """
        self.annas_mcp_path = Path(annas_mcp_path)
        self.api_key = api_key
        self.download_path = Path(download_path)
        self.logger = logger or logging.getLogger(__name__)
        
        # Validate setup
        self._validate_setup()
        
        # Download tracking
        self.downloads_today = 0
        self.download_limit = 25  # Brilliant Bookworm membership limit
        
    def _validate_setup(self):
        """Validate that the annas-mcp tool and paths are properly configured"""
        if not self.annas_mcp_path.exists():
            raise FileNotFoundError(f"annas-mcp binary not found at {self.annas_mcp_path}")
            
        if not self.annas_mcp_path.is_file():
            raise ValueError(f"annas-mcp path is not a file: {self.annas_mcp_path}")
            
        # Check if binary is executable
        if not os.access(self.annas_mcp_path, os.X_OK):
            raise PermissionError(f"annas-mcp binary is not executable: {self.annas_mcp_path}")
            
        # Create download directory if it doesn't exist
        self.download_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"AnnaArchiveClient initialized with download path: {self.download_path}")
    
    def search_book(self, author: str, title: str, max_results: int = 5) -> List[BookSearchResult]:
        """
        Search for a book by author and title
        
        Args:
            author: Author name
            title: Book title
            max_results: Maximum number of results to return
            
        Returns:
            List of BookSearchResult objects
        """
        query = f"{author} {title}".strip()
        self.logger.info(f"Searching for: {query}")
        
        try:
            # Set environment variables for the command
            env = os.environ.copy()
            env['ANNAS_SECRET_KEY'] = self.api_key
            env['ANNAS_DOWNLOAD_PATH'] = str(self.download_path)
            
            # Run search command
            cmd = [str(self.annas_mcp_path), "search", query]
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                self.logger.error(f"Search failed: {result.stderr}")
                return []
                
            # Parse search results (this will need to be adapted based on actual output format)
            search_results = self._parse_search_output(result.stdout)
            return search_results[:max_results]
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"Search timeout for query: {query}")
            return []
        except Exception as e:
            self.logger.error(f"Search error for query '{query}': {str(e)}")
            return []
    
    def _parse_search_output(self, output: str) -> List[BookSearchResult]:
        """
        Parse the output from annas-mcp search command
        
        Note: This method will need to be updated based on the actual
        output format of the annas-mcp search command
        """
        results = []
        
        # For now, return a placeholder - we'll need to see actual output format
        # to implement proper parsing
        lines = output.strip().split('\n')
        
        # Log the raw output for debugging
        self.logger.debug(f"Raw search output:\n{output}")
        
        # TODO: Implement actual parsing based on annas-mcp output format
        # This is a placeholder that will be updated after seeing real output
        
        return results
    
    def download_book(self, md5: str, filename: Optional[str] = None) -> DownloadResult:
        """
        Download a book by its MD5 hash
        
        Args:
            md5: MD5 hash of the book to download
            filename: Optional custom filename for download
            
        Returns:
            DownloadResult object with download information
        """
        if self.downloads_today >= self.download_limit:
            error_msg = f"Download limit reached ({self.download_limit}/day)"
            self.logger.warning(error_msg)
            return DownloadResult(success=False, md5=md5, error_message=error_msg)
        
        self.logger.info(f"Downloading book with MD5: {md5}")
        start_time = time.time()
        
        try:
            # Set environment variables
            env = os.environ.copy()
            env['ANNAS_SECRET_KEY'] = self.api_key
            env['ANNAS_DOWNLOAD_PATH'] = str(self.download_path)
            
            # Prepare command
            cmd = [str(self.annas_mcp_path), "download", md5]
            if filename:
                cmd.append(filename)
            
            # Execute download
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout for downloads
            )
            
            download_time = time.time() - start_time
            
            if result.returncode == 0:
                self.downloads_today += 1
                
                # Try to find the downloaded file
                downloaded_file = self._find_downloaded_file(md5, filename)
                
                self.logger.info(f"Download successful: {md5} in {download_time:.1f}s")
                return DownloadResult(
                    success=True,
                    md5=md5,
                    filepath=downloaded_file,
                    download_time=download_time
                )
            else:
                error_msg = f"Download failed: {result.stderr}"
                self.logger.error(error_msg)
                return DownloadResult(
                    success=False,
                    md5=md5,
                    error_message=error_msg,
                    download_time=download_time
                )
                
        except subprocess.TimeoutExpired:
            error_msg = f"Download timeout for MD5: {md5}"
            self.logger.error(error_msg)
            return DownloadResult(success=False, md5=md5, error_message=error_msg)
        except Exception as e:
            error_msg = f"Download error for MD5 '{md5}': {str(e)}"
            self.logger.error(error_msg)
            return DownloadResult(success=False, md5=md5, error_message=error_msg)
    
    def _find_downloaded_file(self, md5: str, filename: Optional[str]) -> Optional[Path]:
        """Find the downloaded file in the download directory"""
        # Check for files with the MD5 in the name or the specified filename
        for file_path in self.download_path.iterdir():
            if file_path.is_file():
                if md5 in file_path.name or (filename and filename in file_path.name):
                    return file_path
        return None
    
    def get_download_status(self) -> Dict[str, Any]:
        """Get current download status and limits"""
        return {
            "downloads_today": self.downloads_today,
            "download_limit": self.download_limit,
            "remaining_downloads": self.download_limit - self.downloads_today,
            "download_path": str(self.download_path)
        }


class CorpusCreationManager:
    """Manager for creating romance novel corpus from CSV metadata"""
    
    def __init__(self, 
                 client: AnnaArchiveClient,
                 metadata_csv_path: str,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the corpus creation manager
        
        Args:
            client: Configured AnnaArchiveClient
            metadata_csv_path: Path to CSV with book metadata
            logger: Optional logger instance
        """
        self.client = client
        self.metadata_csv_path = Path(metadata_csv_path)
        self.logger = logger or logging.getLogger(__name__)
        
        # Tracking
        self.search_results = []
        self.download_results = []
        self.failed_searches = []
        self.failed_downloads = []
        
    def load_book_metadata(self) -> List[Dict[str, Any]]:
        """Load book metadata from CSV file"""
        books = []
        
        try:
            with open(self.metadata_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['title'] and row['author_name']:  # Skip empty rows
                        books.append(row)
                        
        except Exception as e:
            self.logger.error(f"Error loading metadata CSV: {str(e)}")
            raise
            
        self.logger.info(f"Loaded {len(books)} books from metadata CSV")
        return books
    
    def search_all_books(self, books: List[Dict[str, Any]]) -> List[Tuple[Dict, List[BookSearchResult]]]:
        """
        Search for all books in the metadata list
        
        Args:
            books: List of book metadata dictionaries
            
        Returns:
            List of tuples (metadata, search_results)
        """
        all_results = []
        
        for i, book in enumerate(books, 1):
            self.logger.info(f"Searching book {i}/{len(books)}: {book['title']} by {book['author_name']}")
            
            try:
                search_results = self.client.search_book(
                    author=book['author_name'],
                    title=book['title']
                )
                
                if search_results:
                    all_results.append((book, search_results))
                    self.search_results.append((book, search_results))
                else:
                    self.failed_searches.append(book)
                    self.logger.warning(f"No results found for: {book['title']} by {book['author_name']}")
                    
                # Small delay to be respectful to the API
                time.sleep(1)
                
            except Exception as e:
                self.failed_searches.append(book)
                self.logger.error(f"Search error for {book['title']}: {str(e)}")
                
        self.logger.info(f"Search complete: {len(all_results)} successful, {len(self.failed_searches)} failed")
        return all_results
    
    def save_search_metadata(self, 
                           search_results: List[Tuple[Dict, List[BookSearchResult]]],
                           output_path: str):
        """Save search results metadata to JSON file"""
        metadata = {
            "timestamp": time.time(),
            "total_searches": len(search_results) + len(self.failed_searches),
            "successful_searches": len(search_results),
            "failed_searches": len(self.failed_searches),
            "results": []
        }
        
        for book_meta, results in search_results:
            result_data = {
                "goodreads_metadata": book_meta,
                "anna_archive_results": [
                    {
                        "title": r.title,
                        "author": r.author,
                        "md5": r.md5,
                        "extension": r.extension,
                        "filesize": r.filesize,
                        "language": r.language,
                        "year": r.year,
                        "publisher": r.publisher
                    } for r in results
                ]
            }
            metadata["results"].append(result_data)
        
        # Save failed searches
        metadata["failed_searches"] = self.failed_searches
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"Search metadata saved to: {output_path}")


def setup_logging(log_dir: str, level: str = "INFO") -> logging.Logger:
    """Set up logging for corpus creation"""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped log file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"corpus_creation_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("corpus_creation")
    logger.info(f"Logging initialized: {log_file}")
    return logger


def load_environment_config(config_path: str) -> Dict[str, str]:
    """Load environment configuration from file"""
    config = {}
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()
    
    return config


if __name__ == "__main__":
    # Example usage for testing
    print("Anna's Archive Client for Romance Novel Corpus Creation")
    print("This module should be imported and used with proper configuration.")
