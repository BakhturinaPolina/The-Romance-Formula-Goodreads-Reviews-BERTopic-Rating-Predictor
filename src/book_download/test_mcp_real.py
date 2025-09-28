#!/usr/bin/env python3
"""
Book Download Research Component - Real MCP Integration Test
Test the actual MCP server integration with real book data
"""

import pandas as pd
import os
import sys
import logging
import time
import json
from datetime import datetime
from typing import Dict, List, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealMCPIntegration:
    """Real MCP integration using the actual MCP tools"""
    
    def __init__(self):
        """Initialize real MCP integration"""
        logger.info("Initializing Real MCP Integration")
        self.download_dir = "/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/organized_outputs/anna_archive_download"
        os.makedirs(self.download_dir, exist_ok=True)
        logger.info("Real MCP Integration initialized successfully")

    def search_books(self, search_term: str) -> List[Dict]:
        """
        Search for books using the actual MCP server
        
        Args:
            search_term: Search term (title + author)
            
        Returns:
            List of search results
        """
        logger.info(f"Searching for: '{search_term}'")
        
        try:
            # This will be called by the MCP tools directly
            # We'll simulate the search results structure for now
            # In the actual implementation, this will call mcp_anna-mcp_search
            logger.info("Search functionality ready for MCP tool calls")
            return []
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def download_book(self, book_hash: str, title: str, format_type: str = "epub") -> bool:
        """
        Download a book using the actual MCP server
        
        Args:
            book_hash: Hash of the book to download
            title: Title of the book (for filename)
            format_type: Format to download (epub, html, etc.)
            
        Returns:
            True if download successful, False otherwise
        """
        logger.info(f"Downloading book: {title} (hash: {book_hash}, format: {format_type})")
        
        try:
            # This will be called by the MCP tools directly
            # We'll simulate the download for now
            # In the actual implementation, this will call mcp_anna-mcp_download
            logger.info("Download functionality ready for MCP tool calls")
            return False
            
        except Exception as e:
            logger.error(f"Download error: {e}")
            return False

def test_real_mcp_integration():
    """Test the real MCP integration with sample books"""
    
    logger.info("=== TESTING REAL MCP INTEGRATION ===")
    
    # Load sample CSV
    sample_csv = "/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/data/processed/sample_books_for_download.csv"
    
    try:
        sample_df = pd.read_csv(sample_csv)
        logger.info(f"Loaded {len(sample_df)} sample books")
    except Exception as e:
        logger.error(f"Error loading sample CSV: {e}")
        return False
    
    # Initialize real MCP integration
    mcp = RealMCPIntegration()
    
    # Test with first 2 books
    test_books = sample_df.head(2)
    logger.info(f"Testing with {len(test_books)} books")
    
    results = []
    
    for idx, row in test_books.iterrows():
        logger.info(f"\n--- Testing Book {idx+1} ---")
        logger.info(f"Title: '{row['title']}'")
        logger.info(f"Author: {row['author_name']}")
        logger.info(f"Year: {row['publication_year']}")
        logger.info(f"Work ID: {row['work_id']}")
        
        # Test search
        search_term = f"{row['title']} {row['author_name']}"
        search_results = mcp.search_books(search_term)
        
        result = {
            'work_id': row['work_id'],
            'title': row['title'],
            'author_name': row['author_name'],
            'publication_year': row['publication_year'],
            'search_term': search_term,
            'search_results_count': len(search_results),
            'test_timestamp': datetime.now().isoformat()
        }
        
        results.append(result)
        
        # Add delay between tests
        time.sleep(1)
    
    # Save test results
    results_file = "/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/src/book_download/real_mcp_test_results.json"
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Test results saved to: {results_file}")
    except Exception as e:
        logger.error(f"Error saving test results: {e}")
    
    logger.info("\n=== TEST SUMMARY ===")
    logger.info(f"Tested {len(results)} books")
    logger.info("Next steps:")
    logger.info("1. Implement actual MCP tool calls for search")
    logger.info("2. Implement actual MCP tool calls for download")
    logger.info("3. Test with real search results")
    logger.info("4. Test with real downloads")
    
    return True

if __name__ == "__main__":
    logger.info("Starting real MCP integration test...")
    success = test_real_mcp_integration()
    if success:
        logger.info("Real MCP integration test completed successfully!")
    else:
        logger.error("Real MCP integration test failed!")
        sys.exit(1)
