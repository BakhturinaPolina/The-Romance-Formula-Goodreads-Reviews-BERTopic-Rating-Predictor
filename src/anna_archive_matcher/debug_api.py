#!/usr/bin/env python3
"""
Debug Anna's Archive API calls to understand search behavior
"""

import sys
import logging
import json
from pathlib import Path

# Add the book_download directory to the path
sys.path.append(str((Path(__file__).parent.parent / "book_download").resolve()))

from anna_api_client import AnnaAPIClient

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("debug_api")

# Your API key
API_KEY = "BvQiX8XK9Y2LNp5yuXXnV2yqyRPWP"

def debug_api_search():
    """Debug the API search to see what's happening"""
    
    logger.info("Anna's Archive API Debug")
    logger.info("=" * 40)
    
    # Initialize the API client
    client = AnnaAPIClient(
        api_key=API_KEY,
        use_tor=True,
        timeout=30
    )
    
    # Test with a simple search
    test_title = "Pride and Prejudice"
    test_author = "Jane Austen"
    
    logger.info(f"Testing search: '{test_title}' by {test_author}")
    
    # Manually make the API call to see the raw response
    search_term = f"{test_title} {test_author}".strip()
    params = {"query": search_term, "key": API_KEY, "page": 1, "type": "books"}
    
    logger.info(f"Search term: '{search_term}'")
    logger.info(f"Params: {params}")
    
    for mirror in client.MIRRORS:
        url = f"{mirror}/dyn/api/search.json"
        logger.info(f"Trying mirror: {url}")
        
        try:
            response = client._make_tor_request(url, params, client.headers)
            logger.info(f"Response status: {response.status_code}")
            logger.info(f"Response headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    logger.info(f"Response data type: {type(data)}")
                    logger.info(f"Response data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                    
                    if isinstance(data, dict):
                        results = data.get("results", [])
                        logger.info(f"Number of results: {len(results)}")
                        
                        if results:
                            logger.info("First result:")
                            logger.info(json.dumps(results[0], indent=2))
                        else:
                            logger.info("No results found")
                    else:
                        logger.info(f"Raw response: {data}")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                    logger.info(f"Raw response text: {response.text[:500]}")
            else:
                logger.error(f"HTTP error: {response.status_code}")
                logger.info(f"Response text: {response.text[:500]}")
                
        except Exception as e:
            logger.error(f"Request failed: {e}")
        
        logger.info("-" * 40)

if __name__ == "__main__":
    debug_api_search()
