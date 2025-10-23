#!/usr/bin/env python3
"""
Hybrid Anna's Archive Search and Download System
Combines web scraping for search with API for fast downloads
"""

import sys
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

# Add the utils and book_download directories to the path
sys.path.append(str((Path(__file__).parent / "utils").resolve()))
sys.path.append(str((Path(__file__).parent.parent / "book_download").resolve()))

from proxy_automated_search import ProxyAutomatedSearcher
from anna_api_client import AnnaAPIClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("hybrid_search_download")

class HybridAnnaSearcher:
    """
    Hybrid searcher that combines web scraping for search with API for downloads
    """
    
    def __init__(self, api_key: str, proxy_config: Optional[Dict] = None):
        """
        Initialize the hybrid searcher
        
        Args:
            api_key: Anna's Archive member API key
            proxy_config: Proxy configuration for web scraping
        """
        self.api_key = api_key
        
        # Initialize web scraper for search
        self.web_searcher = ProxyAutomatedSearcher(
            delay_range=(2.0, 4.0),
            proxy_config=proxy_config or {'type': 'tor', 'host': '127.0.0.1', 'port': 9050}
        )
        
        # Initialize API client for downloads
        self.api_client = AnnaAPIClient(
            api_key=api_key,
            use_tor=True,
            timeout=30
        )
        
        logger.info("Hybrid Anna's Archive searcher initialized")
        logger.info(f"Web scraper: {len(self.web_searcher.base_urls)} domains")
        logger.info(f"API client: {len(self.api_client.MIRRORS)} mirrors")
    
    def search_and_download(self, title: str, author: str, max_retries: int = 2) -> Optional[Dict]:
        """
        Search for a book and get download information
        
        Args:
            title: Book title
            author: Author name
            max_retries: Maximum retry attempts for search
            
        Returns:
            Dictionary with book info and download URL, or None if not found
        """
        logger.info(f"Searching: '{title}' by {author}")
        
        # Step 1: Use web scraping to find the book and get MD5
        try:
            search_result = self.web_searcher.search_book(title, author, max_retries)
            
            if not search_result:
                logger.info(f"❌ Not found via web search: {title}")
                return None
            
            # Extract MD5 from the search result
            md5 = search_result.get('md5')
            if not md5:
                logger.warning(f"⚠️  No MD5 found in search result for: {title}")
                return None
            
            logger.info(f"✅ Found via web search: {title} (MD5: {md5})")
            
        except Exception as e:
            logger.error(f"❌ Web search failed for '{title}': {e}")
            return None
        
        # Step 2: Use API to get fast download URL
        try:
            download_url = self.api_client.get_fast_download(md5)
            
            if not download_url:
                logger.warning(f"⚠️  No download URL available for: {title}")
                return None
            
            logger.info(f"✅ Got download URL for: {title}")
            
            # Combine results
            result = {
                'title': title,
                'author': author,
                'md5': md5,
                'download_url': download_url,
                'search_result': search_result
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ API download failed for '{title}': {e}")
            return None
    
    def search_books_batch(self, books_df: pd.DataFrame, max_books: int = None) -> List[Dict]:
        """
        Search multiple books in batch
        
        Args:
            books_df: DataFrame with 'title' and 'author_name' columns
            max_books: Maximum number of books to process
            
        Returns:
            List of successful search results
        """
        if max_books:
            books_df = books_df.head(max_books)
        
        results = []
        
        for idx, book in books_df.iterrows():
            logger.info(f"Processing book {idx + 1}/{len(books_df)}")
            
            result = self.search_and_download(
                title=book['title'],
                author=book['author_name']
            )
            
            if result:
                results.append(result)
                logger.info(f"✅ Success: {book['title']}")
            else:
                logger.info(f"❌ Failed: {book['title']}")
            
            logger.info("")
        
        logger.info(f"Batch complete: {len(results)}/{len(books_df)} books found")
        return results

def test_hybrid_system():
    """Test the hybrid search and download system"""
    
    logger.info("Hybrid Anna's Archive Search & Download Test")
    logger.info("=" * 50)
    
    # Your API key
    API_KEY = "BvQiX8XK9Y2LNp5yuXXnV2yqyRPWP"
    
    # Initialize hybrid searcher
    searcher = HybridAnnaSearcher(API_KEY)
    
    # Test with a few books from our dataset
    csv_path = Path("utils/priority_lists/test_sample_50_books.csv")
    if not csv_path.exists():
        logger.error(f"Test CSV not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} books from test sample")
    
    # Test with first 3 books
    test_books = df.head(3)
    
    results = searcher.search_books_batch(test_books)
    
    # Summary
    logger.info("=" * 50)
    logger.info(f"Test Results: {len(results)}/{len(test_books)} books found")
    
    for result in results:
        logger.info(f"✅ {result['title']} by {result['author']}")
        logger.info(f"   MD5: {result['md5']}")
        logger.info(f"   Download: {result['download_url'][:60]}...")
    
    return len(results) > 0

if __name__ == "__main__":
    success = test_hybrid_system()
    sys.exit(0 if success else 1)
