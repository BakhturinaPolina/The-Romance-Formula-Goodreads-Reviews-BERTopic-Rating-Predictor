#!/usr/bin/env python3
"""
Targeted Anna's Archive Dataset Downloader
Downloads only romance/fiction books in English without torrents
"""

import requests
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TargetedDownloader:
    """
    Download specific romance/fiction books from Anna's Archive
    without downloading full datasets
    """
    
    def __init__(self, output_dir: str = "targeted_downloads"):
        """
        Initialize the targeted downloader
        
        Args:
            output_dir: Directory to save downloaded books
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Anna's Archive search API endpoints
        self.search_url = "https://annas-archive.org/search"
        self.download_url = "https://annas-archive.org/md5"
        
        # Romance/fiction keywords for filtering
        self.romance_keywords = [
            "romance", "love story", "romantic", "chick lit", 
            "contemporary romance", "historical romance", "paranormal romance"
        ]
        
        # Fiction genres to include
        self.fiction_genres = [
            "fiction", "novel", "romance", "contemporary", "historical",
            "paranormal", "fantasy", "mystery", "thriller", "drama"
        ]
        
        logger.info(f"Targeted downloader initialized. Output: {self.output_dir}")
    
    def search_romance_books(self, query: str, limit: int = 100) -> List[Dict]:
        """
        Search for romance books using Anna's Archive search
        
        Args:
            query: Search query (title, author, or ISBN)
            limit: Maximum number of results
            
        Returns:
            List of book metadata
        """
        logger.info(f"Searching for: {query}")
        
        # Search parameters for romance/fiction
        params = {
            'q': f"{query} romance fiction",
            'content': 'book',
            'language': 'en',
            'extension': 'epub,pdf',
            'limit': limit
        }
        
        try:
            response = requests.get(self.search_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse search results (this would need to be adapted based on actual API response)
            results = self._parse_search_results(response.text)
            
            # Filter for romance/fiction
            filtered_results = self._filter_romance_fiction(results)
            
            logger.info(f"Found {len(filtered_results)} romance/fiction books")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Search failed for '{query}': {e}")
            return []
    
    def download_book(self, book_metadata: Dict) -> Optional[Path]:
        """
        Download a single book
        
        Args:
            book_metadata: Book metadata with MD5 hash
            
        Returns:
            Path to downloaded file or None if failed
        """
        md5_hash = book_metadata.get('md5')
        if not md5_hash:
            logger.warning("No MD5 hash found for book")
            return None
        
        try:
            # Construct download URL
            download_url = f"{self.download_url}/{md5_hash}"
            
            # Download the file
            response = requests.get(download_url, stream=True, timeout=60)
            response.raise_for_status()
            
            # Determine file extension
            content_type = response.headers.get('content-type', '')
            if 'epub' in content_type:
                ext = 'epub'
            elif 'pdf' in content_type:
                ext = 'pdf'
            else:
                ext = 'unknown'
            
            # Create filename
            title = book_metadata.get('title', 'unknown')
            author = book_metadata.get('author', 'unknown')
            filename = f"{title}_{author}_{md5_hash[:8]}.{ext}"
            filename = self._sanitize_filename(filename)
            
            file_path = self.output_dir / filename
            
            # Save file
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded: {filename}")
            return file_path
            
        except Exception as e:
            logger.error(f"Download failed for {book_metadata.get('title', 'unknown')}: {e}")
            return None
    
    def batch_download_from_csv(self, csv_path: str, max_books: int = 1000) -> List[Path]:
        """
        Download books from your romance CSV file
        
        Args:
            csv_path: Path to your romance books CSV
            max_books: Maximum number of books to download
            
        Returns:
            List of downloaded file paths
        """
        import pandas as pd
        
        logger.info(f"Loading romance books from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Limit to max_books for testing
        if max_books:
            df = df.head(max_books)
        
        downloaded_files = []
        
        for idx, book in tqdm(df.iterrows(), total=len(df), desc="Downloading books"):
            if idx % 10 == 0:
                logger.info(f"Processing book {idx}/{len(df)}")
            
            # Search for the book
            query = f"{book['title']} {book['author_name']}"
            search_results = self.search_romance_books(query, limit=5)
            
            if search_results:
                # Try to download the first match
                downloaded_file = self.download_book(search_results[0])
                if downloaded_file:
                    downloaded_files.append(downloaded_file)
            
            # Rate limiting
            time.sleep(1)
        
        logger.info(f"Downloaded {len(downloaded_files)} books")
        return downloaded_files
    
    def _parse_search_results(self, html_content: str) -> List[Dict]:
        """
        Parse search results from Anna's Archive HTML response
        
        Args:
            html_content: HTML content from search response
            
        Returns:
            List of book metadata dictionaries
        """
        # This is a simplified parser - you'd need to adapt based on actual HTML structure
        results = []
        
        # For now, return empty list - this would need proper HTML parsing
        # or API integration based on Anna's Archive's actual response format
        
        return results
    
    def _filter_romance_fiction(self, results: List[Dict]) -> List[Dict]:
        """
        Filter results to only include romance/fiction books
        
        Args:
            results: List of book metadata
            
        Returns:
            Filtered list of romance/fiction books
        """
        filtered = []
        
        for book in results:
            title = book.get('title', '').lower()
            description = book.get('description', '').lower()
            genres = book.get('genres', '').lower()
            
            # Check if it's romance/fiction
            is_romance = any(keyword in title or keyword in description or keyword in genres 
                           for keyword in self.romance_keywords)
            
            is_fiction = any(genre in title or genre in description or genre in genres 
                           for genre in self.fiction_genres)
            
            if is_romance or is_fiction:
                filtered.append(book)
        
        return filtered
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename for safe saving
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        import re
        
        # Remove invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Limit length
        if len(filename) > 200:
            filename = filename[:200]
        
        return filename


def main():
    """
    Main function for targeted downloading
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Targeted Anna Archive Downloader')
    parser.add_argument('--romance-csv', required=True,
                       help='Path to romance books CSV file')
    parser.add_argument('--max-books', type=int, default=100,
                       help='Maximum number of books to download')
    parser.add_argument('--output-dir', default='targeted_downloads',
                       help='Output directory for downloads')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize downloader
    downloader = TargetedDownloader(args.output_dir)
    
    # Download books
    downloaded_files = downloader.batch_download_from_csv(
        args.romance_csv, 
        args.max_books
    )
    
    print(f"Downloaded {len(downloaded_files)} books to {args.output_dir}")


if __name__ == "__main__":
    main()
