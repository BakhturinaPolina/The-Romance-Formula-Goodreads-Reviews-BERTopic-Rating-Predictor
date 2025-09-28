#!/usr/bin/env python3
"""
Book Download Research Component - Test Download
Simple script to test anna-mcp server functionality with sample books
"""

import pandas as pd
import os
import sys
import logging
import time
import json
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_anna_mcp_search(title, author_name):
    """Test searching for a book using anna-mcp server"""
    logger.info(f"Testing search for: '{title}' by {author_name}")
    
    # Create search term combining title and author
    search_term = f"{title} {author_name}"
    logger.info(f"Search term: '{search_term}'")
    
    # For now, we'll simulate the search since we need to test the MCP integration
    # In the actual implementation, this will call the anna-mcp server
    logger.info("Search functionality will be implemented with MCP server calls")
    
    return {
        'search_term': search_term,
        'title': title,
        'author': author_name,
        'status': 'search_simulated'
    }

def test_download_workflow():
    """Test the complete download workflow with sample books"""
    
    # Load sample CSV
    sample_csv = "/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/data/processed/sample_books_for_download.csv"
    
    logger.info("=== TESTING DOWNLOAD WORKFLOW ===")
    logger.info(f"Loading sample CSV: {sample_csv}")
    
    try:
        sample_df = pd.read_csv(sample_csv)
        logger.info(f"Loaded {len(sample_df)} sample books")
    except Exception as e:
        logger.error(f"Error loading sample CSV: {e}")
        return False
    
    # Test with first 2-3 books
    test_books = sample_df.head(3)
    logger.info(f"Testing with {len(test_books)} books")
    
    results = []
    
    for idx, row in test_books.iterrows():
        logger.info(f"\n--- Testing Book {idx+1} ---")
        logger.info(f"Title: '{row['title']}'")
        logger.info(f"Author: {row['author_name']}")
        logger.info(f"Year: {row['publication_year']}")
        logger.info(f"Work ID: {row['work_id']}")
        
        # Test search
        search_result = test_anna_mcp_search(row['title'], row['author_name'])
        results.append({
            'work_id': row['work_id'],
            'title': row['title'],
            'author_name': row['author_name'],
            'publication_year': row['publication_year'],
            'search_result': search_result,
            'test_timestamp': datetime.now().isoformat()
        })
        
        # Add delay between tests
        time.sleep(1)
    
    # Save test results
    results_file = "/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/src/book_download/test_results.json"
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Test results saved to: {results_file}")
    except Exception as e:
        logger.error(f"Error saving test results: {e}")
    
    logger.info("\n=== TEST SUMMARY ===")
    logger.info(f"Tested {len(results)} books")
    logger.info("Next steps:")
    logger.info("1. Implement actual MCP server calls for search")
    logger.info("2. Implement download functionality")
    logger.info("3. Add progress tracking system")
    logger.info("4. Add rate limiting (999 books per day)")
    
    return True

if __name__ == "__main__":
    logger.info("Starting download workflow test...")
    success = test_download_workflow()
    if success:
        logger.info("Download workflow test completed successfully!")
    else:
        logger.error("Download workflow test failed!")
        sys.exit(1)
