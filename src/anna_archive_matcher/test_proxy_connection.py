#!/usr/bin/env python3
"""
Test Proxy Connection for Anna's Archive
Simple script to test if proxy is working before running full automation
"""

import sys
import logging
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent / "utils"))

from proxy_automated_search import ProxyAutomatedSearcher

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_proxy_connection(proxy_type: str = "tor", 
                         proxy_host: str = "127.0.0.1", 
                         proxy_port: int = 9050):
    """
    Test proxy connection
    
    Args:
        proxy_type: Type of proxy (tor, socks5, http)
        proxy_host: Proxy host
        proxy_port: Proxy port
    """
    logger.info(f"Testing {proxy_type} proxy connection...")
    logger.info(f"Host: {proxy_host}, Port: {proxy_port}")
    
    # Configure proxy
    proxy_config = {
        'type': proxy_type,
        'host': proxy_host,
        'port': proxy_port
    }
    
    try:
        # Initialize searcher
        searcher = ProxyAutomatedSearcher(proxy_config=proxy_config)
        
        # Test connection
        if searcher.test_proxy_connection():
            logger.info("✅ Proxy connection successful!")
            
            # Test a simple search
            logger.info("Testing search functionality...")
            result = searcher.search_book("test", "test", max_retries=1)
            
            if result:
                logger.info("✅ Search functionality working!")
            else:
                logger.info("⚠️  Search test completed (no results found, but connection works)")
            
            return True
        else:
            logger.error("❌ Proxy connection failed!")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing proxy: {e}")
        return False


def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Proxy Connection')
    parser.add_argument('--proxy-type', choices=['tor', 'socks5', 'http'], default='tor',
                       help='Proxy type to test')
    parser.add_argument('--proxy-host', default='127.0.0.1',
                       help='Proxy host')
    parser.add_argument('--proxy-port', type=int, default=9050,
                       help='Proxy port')
    
    args = parser.parse_args()
    
    logger.info("Anna's Archive Proxy Connection Test")
    logger.info("=" * 40)
    
    success = test_proxy_connection(
        proxy_type=args.proxy_type,
        proxy_host=args.proxy_host,
        proxy_port=args.proxy_port
    )
    
    if success:
        logger.info("\n🎉 Proxy is working! You can now run the full automation.")
        logger.info("\nNext steps:")
        logger.info("1. Run: python utils/proxy_automated_search.py --help")
        logger.info("2. Start with a small test: --max-books 5")
        logger.info("3. Scale up to larger batches once confirmed working")
    else:
        logger.error("\n❌ Proxy connection failed. Please check:")
        logger.error("1. Is Tor/VPN running?")
        logger.error("2. Are the host/port correct?")
        logger.error("3. Check firewall settings")
        logger.error("4. See PROXY_SETUP_GUIDE.md for detailed instructions")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
