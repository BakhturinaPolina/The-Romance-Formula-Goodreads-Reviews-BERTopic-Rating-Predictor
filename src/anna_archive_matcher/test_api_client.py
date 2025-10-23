#!/usr/bin/env python3
"""
Test Anna's Archive API Client with member API key
Tests the official API client to bypass 403 blocking issues
"""

import sys
import logging
from pathlib import Path

# Add the book_download directory to the path
sys.path.append(str((Path(__file__).parent.parent / "book_download").resolve()))

from anna_api_client import AnnaAPIClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_api_client")

# Your API key
API_KEY = "BvQiX8XK9Y2LNp5yuXXnV2yqyRPWP"

# Test books (classics that should be available)
TEST_BOOKS = [
    ("Pride and Prejudice", "Jane Austen"),
    ("Jane Eyre", "Charlotte Brontë"),
    ("Wuthering Heights", "Emily Brontë"),
    ("Dracula", "Bram Stoker"),
    ("Moby-Dick", "Herman Melville"),
    ("The Picture of Dorian Gray", "Oscar Wilde"),
    ("Frankenstein", "Mary Shelley"),
    ("The Adventures of Sherlock Holmes", "Arthur Conan Doyle"),
]

def test_api_client():
    """Test the Anna's Archive API client with member API key"""
    
    logger.info("Anna's Archive API Client Test")
    logger.info("=" * 40)
    logger.info(f"API Key: {API_KEY[:8]}...{API_KEY[-4:]}")
    logger.info("")
    
    # Initialize the API client
    try:
        client = AnnaAPIClient(
            api_key=API_KEY,
            use_tor=True,  # Still use Tor for additional privacy
            timeout=30
        )
        logger.info("✅ API client initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize API client: {e}")
        return False
    
    # Test API connectivity
    logger.info("Testing API connectivity...")
    try:
        # Try a simple search to test connectivity
        test_md5 = client.search_md5("test", "test")
        if test_md5:
            logger.info("✅ API connectivity test successful")
        else:
            logger.info("⚠️  API connectivity test completed (no results for test search)")
    except Exception as e:
        logger.error(f"❌ API connectivity test failed: {e}")
        return False
    
    # Test with classic books
    logger.info(f"Testing with {len(TEST_BOOKS)} classic books...")
    logger.info("")
    
    found_books = 0
    
    for i, (title, author) in enumerate(TEST_BOOKS, 1):
        logger.info(f"[{i}/{len(TEST_BOOKS)}] Searching: '{title}' by {author}")
        
        try:
            # Search for the book using the API
            md5 = client.search_md5(title, author, prefer_exts=("epub", "pdf"))
            
            if md5:
                logger.info(f"✅ Found: {title}")
                logger.info(f"   MD5: {md5}")
                
                # Test getting download URL
                try:
                    download_url = client.get_fast_download(md5)
                    if download_url:
                        logger.info(f"   Download URL: {download_url[:60]}...")
                        found_books += 1
                    else:
                        logger.warning(f"   ⚠️  No download URL available")
                except Exception as e:
                    logger.warning(f"   ⚠️  Download URL test failed: {e}")
            else:
                logger.info(f"❌ Not found: {title}")
                
        except Exception as e:
            logger.error(f"❌ Search failed for '{title}': {e}")
        
        logger.info("")
    
    # Summary
    logger.info("=" * 40)
    logger.info(f"Results: {found_books}/{len(TEST_BOOKS)} books found with download URLs")
    
    if found_books > 0:
        logger.info("🎉 API client is working! You can now use it for automated downloads.")
        return True
    else:
        logger.info("⚠️  No books found. This might indicate:")
        logger.info("1. API key issues")
        logger.info("2. Tor connectivity problems")
        logger.info("3. Search terms need adjustment")
        return False

if __name__ == "__main__":
    success = test_api_client()
    sys.exit(0 if success else 1)
