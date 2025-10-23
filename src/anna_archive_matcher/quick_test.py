#!/usr/bin/env python3
"""
Quick test script for proxy search functionality
Tests with a single book and shorter timeouts
"""

import sys
import logging
from pathlib import Path

# Add the utils directory to the path
sys.path.append(str(Path(__file__).parent / "utils"))

from proxy_automated_search import ProxyAutomatedSearcher
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_test():
    """Run a quick test with a single book"""
    
    # Load test data
    csv_path = Path("utils/priority_lists/test_sample_50_books.csv")
    if not csv_path.exists():
        logger.error(f"Test CSV not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} books from test sample")
    
    # Create searcher with shorter timeouts for testing
    searcher = ProxyAutomatedSearcher(
        delay_range=(1.0, 2.0),  # Shorter delays
        proxy_config={
            'type': 'tor',
            'host': '127.0.0.1',
            'port': 9050
        }
    )
    
    # Test proxy connection
    logger.info("Testing proxy connection...")
    if not searcher.test_proxy_connection():
        logger.error("Proxy connection failed!")
        return
    
    logger.info("✅ Proxy connection successful!")
    
    # Try classical books that are more likely to be available
    classical_test_books = [
        {'title': 'Pride and Prejudice', 'author_name': 'Jane Austen'},
        {'title': 'Romeo and Juliet', 'author_name': 'William Shakespeare'},
        {'title': 'The Great Gatsby', 'author_name': 'F. Scott Fitzgerald'},
        {'title': '1984', 'author_name': 'George Orwell'},
        {'title': 'To Kill a Mockingbird', 'author_name': 'Harper Lee'},
    ]
    
    # Also try a few from our romance dataset
    romance_test_books = [
        df.iloc[0],  # A Wild Affair by Gemma Townley
        df.iloc[1],  # Aboard the Wishing Star by Debra Parmley  
        df.iloc[2],  # Before You Break by K.C. Wells
    ]
    
    all_test_books = classical_test_books + romance_test_books
    
    for i, test_book in enumerate(all_test_books):
        logger.info(f"Testing book {i+1}/{len(all_test_books)}: '{test_book['title']}' by {test_book['author_name']}")
        
        # Search for this book
        results = searcher.search_book(
            title=test_book['title'],
            author=test_book['author_name'],
            max_retries=1
        )
        
        if results:
            logger.info(f"✅ Found results for '{test_book['title']}'!")
            logger.info(f"Result: {results.get('title', 'Unknown')} - {results.get('url', 'No URL')}")
            logger.info("Quick test completed with successful result!")
            return results
        else:
            logger.info(f"❌ No results for '{test_book['title']}'")
    
    logger.info("No results found for any test books - this might indicate connection issues")
    logger.info("Quick test completed!")

if __name__ == "__main__":
    quick_test()
