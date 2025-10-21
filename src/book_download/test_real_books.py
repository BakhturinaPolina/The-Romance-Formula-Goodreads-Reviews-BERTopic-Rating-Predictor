#!/usr/bin/env python3
"""
Test script for Anna's Archive book downloads using real MD5 hashes
Demonstrates system reliability with actual books from Anna's Archive
"""

import os
import sys
import logging
from typing import List, Tuple, Dict

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from anna_api_client import AnnaAPIClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Set up environment variables for Anna's Archive API"""
    os.environ['ANNAS_SECRET_KEY'] = 'BvQiX8XK9Y2LNp5yuXXnV2yqyRPWP'
    os.environ['ANNAS_DOWNLOAD_PATH'] = '/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/organized_outputs/anna_archive_download'
    
    logger.info("Environment variables set successfully")
    logger.info(f"ANNAS_SECRET_KEY: {os.environ['ANNAS_SECRET_KEY'][:10]}...")
    logger.info(f"ANNAS_DOWNLOAD_PATH: {os.environ['ANNAS_DOWNLOAD_PATH']}")

def get_real_book_md5s() -> List[Tuple[str, str, str]]:
    """
    Get real MD5 hashes for books from Anna's Archive
    Returns list of tuples: (md5, title, author)
    """
    return [
        ('d6e1dc51a50726f00ec438af21952a45', 'Example Book 1', 'Unknown Author'),
        ('4aaa4f9b53b20a0d31aa28fb8c74b7c4', 'Mont-Saint-Michel and Chartres', 'Henry Adams'),
        ('4bde319229eca75f0b7773d0c8319705', 'Equal Danger', 'Leonardo Sciascia'),
        ('81e4ece26ab81e4f9b0ff83e08259066', 'T Zero', 'Italo Calvino'),
    ]

def test_book_downloads():
    """Test downloading books with real MD5 hashes"""
    logger.info("Starting Anna's Archive book download tests")
    logger.info("=" * 60)
    
    # Set up environment
    setup_environment()
    
    # Initialize API client
    client = AnnaAPIClient()
    
    # Get real book MD5s
    books = get_real_book_md5s()
    
    results = []
    
    for md5, title, author in books:
        logger.info(f"Testing: {title} by {author}")
        logger.info(f"MD5: {md5}")
        
        # Create safe filename
        safe_title = ''.join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f'{safe_title}.epub'
        
        try:
            result = client.download_book(md5, filename)
            
            if result.get('success'):
                filepath = result.get('filepath')
                if filepath and os.path.exists(filepath):
                    size = os.path.getsize(filepath)
                    logger.info(f"✅ SUCCESS: Downloaded {size:,} bytes to {filepath}")
                    results.append({
                        'title': title,
                        'author': author,
                        'md5': md5,
                        'success': True,
                        'filepath': filepath,
                        'size': size
                    })
                else:
                    logger.error(f"❌ FAILED: File not found after download")
                    results.append({
                        'title': title,
                        'author': author,
                        'md5': md5,
                        'success': False,
                        'error': 'File not found after download'
                    })
            else:
                logger.error(f"❌ FAILED: {result.get('message', 'Unknown error')}")
                results.append({
                    'title': title,
                    'author': author,
                    'md5': md5,
                    'success': False,
                    'error': result.get('message', 'Unknown error')
                })
                
        except Exception as e:
            logger.error(f"❌ EXCEPTION: {str(e)}")
            results.append({
                'title': title,
                'author': author,
                'md5': md5,
                'success': False,
                'error': str(e)
            })
        
        logger.info("-" * 60)
    
    return results

def print_summary(results: List[Dict]):
    """Print test summary"""
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    logger.info(f"Total tests: {len(results)}")
    logger.info(f"Successful downloads: {len(successful)}")
    logger.info(f"Failed downloads: {len(failed)}")
    logger.info(f"Success rate: {len(successful)/len(results)*100:.1f}%")
    
    if successful:
        logger.info("\n✅ SUCCESSFUL DOWNLOADS:")
        for result in successful:
            logger.info(f"  - {result['title']} by {result['author']} ({result['size']:,} bytes)")
    
    if failed:
        logger.info("\n❌ FAILED DOWNLOADS:")
        for result in failed:
            logger.info(f"  - {result['title']} by {result['author']}: {result['error']}")
    
    logger.info("=" * 60)

def main():
    """Main function"""
    try:
        results = test_book_downloads()
        print_summary(results)
        
        # Return exit code based on results
        if all(r['success'] for r in results):
            logger.info("🎉 All tests passed!")
            return 0
        else:
            logger.warning("⚠️  Some tests failed")
            return 1
            
    except Exception as e:
        logger.error(f"Test script failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
