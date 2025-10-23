#!/usr/bin/env python3
"""
Simple test of Anna's Archive API endpoints
"""

import sys
import logging
import requests
from pathlib import Path

# Add the book_download directory to the path
sys.path.append(str((Path(__file__).parent.parent / "book_download").resolve()))

from anna_api_client import AnnaAPIClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_simple_api")

# Your API key
API_KEY = "BvQiX8XK9Y2LNp5yuXXnV2yqyRPWP"

def test_simple_api():
    """Test simple API endpoints"""
    
    logger.info("Simple Anna's Archive API Test")
    logger.info("=" * 40)
    
    # Initialize the API client
    client = AnnaAPIClient(
        api_key=API_KEY,
        use_tor=True,
        timeout=30
    )
    
    # Test 1: Try the fast download endpoint with a known MD5
    logger.info("Test 1: Testing fast download endpoint")
    known_md5 = "4aaa4f9b53b20a0d31aa28fb8c74b7c4"  # From your existing downloads
    
    try:
        download_url = client.get_fast_download(known_md5)
        if download_url:
            logger.info(f"✅ Fast download works! URL: {download_url[:60]}...")
        else:
            logger.info("❌ Fast download failed")
    except Exception as e:
        logger.error(f"❌ Fast download error: {e}")
    
    logger.info("")
    
    # Test 2: Try different search parameters
    logger.info("Test 2: Testing different search parameters")
    
    # Try without author
    try:
        md5 = client.search_md5("Pride and Prejudice", None)
        if md5:
            logger.info(f"✅ Search without author works! MD5: {md5}")
        else:
            logger.info("❌ Search without author failed")
    except Exception as e:
        logger.error(f"❌ Search without author error: {e}")
    
    # Try with just author
    try:
        md5 = client.search_md5("", "Jane Austen")
        if md5:
            logger.info(f"✅ Search with just author works! MD5: {md5}")
        else:
            logger.info("❌ Search with just author failed")
    except Exception as e:
        logger.error(f"❌ Search with just author error: {e}")
    
    logger.info("")
    
    # Test 3: Try a very simple search
    logger.info("Test 3: Testing very simple search")
    try:
        md5 = client.search_md5("test")
        if md5:
            logger.info(f"✅ Simple search works! MD5: {md5}")
        else:
            logger.info("❌ Simple search failed")
    except Exception as e:
        logger.error(f"❌ Simple search error: {e}")

if __name__ == "__main__":
    test_simple_api()
