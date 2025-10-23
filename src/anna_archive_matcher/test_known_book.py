#!/usr/bin/env python3
"""
Test script using known classics to verify Anna's Archive connectivity
Handles ISP blocking diagnostics and proxy testing
"""

import sys
import logging
import argparse
from pathlib import Path

# Add the utils directory to the path
sys.path.append(str((Path(__file__).parent / "utils").resolve()))

from proxy_automated_search import ProxyAutomatedSearcher

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_known_book")

# Widely available classics that should be in Anna's Archive
CLASSICS = [
    ("Pride and Prejudice", "Jane Austen"),
    ("Jane Eyre", "Charlotte Brontë"),
    ("Wuthering Heights", "Emily Brontë"),
    ("Dracula", "Bram Stoker"),
    ("Moby-Dick", "Herman Melville"),
    ("The Picture of Dorian Gray", "Oscar Wilde"),
    ("Frankenstein", "Mary Shelley"),
    ("The Adventures of Sherlock Holmes", "Arthur Conan Doyle"),
]

def test_known_book(proxy_type="tor", proxy_host="127.0.0.1", proxy_port=9050):
    """
    Test Anna's Archive connectivity using known classics
    
    Args:
        proxy_type: Type of proxy to use ('tor', 'socks5', 'http', 'none')
        proxy_host: Proxy host
        proxy_port: Proxy port
        
    Returns:
        int: Exit code (0=success, 1=proxy failure, 2=no results found)
    """
    proxy_config = None if proxy_type in (None, "none") else {
        "type": proxy_type, 
        "host": proxy_host, 
        "port": proxy_port
    }
    
    searcher = ProxyAutomatedSearcher(
        delay_range=(1.0, 2.0), 
        proxy_config=proxy_config
    )

    if proxy_config:
        logger.info("Testing proxy connection…")
        if not searcher.test_proxy_connection():
            logger.error("❌ Proxy connection failed (likely Tor/VPN not running).")
            logger.error("Make sure Tor is running: sudo systemctl start tor")
            return 1
        logger.info("✅ Proxy connection OK.")
    else:
        logger.info("Running without proxy (will likely hit ISP blocks)")

    logger.info(f"Testing {len(CLASSICS)} classic books...")
    
    for i, (title, author) in enumerate(CLASSICS, 1):
        logger.info(f"[{i}/{len(CLASSICS)}] '{title}' — {author}")
        results = searcher.search_book(title=title, author=author, max_retries=1)
        
        if results:
            logger.info(f"✅ Found: {results.get('title','Unknown')}")
            logger.info(f"   URL: {results.get('url','—')}")
            logger.info("🎉 Anna's Archive is accessible!")
            return 0
        else:
            logger.info(f"❌ Not found: {title}")
    
    logger.info("No results for any test classics.")
    if proxy_config:
        logger.info("This might indicate:")
        logger.info("1. Anna's Archive is down")
        logger.info("2. Search logic needs adjustment")
        logger.info("3. Proxy configuration issues")
    else:
        logger.info("This confirms ISP blocking - use Tor/VPN")
    
    return 2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Anna's Archive connectivity with classic books")
    parser.add_argument("--proxy-type", 
                       choices=["tor", "socks5", "http", "none"], 
                       default="tor",
                       help="Proxy type to use (default: tor)")
    parser.add_argument("--proxy-host", 
                       default="127.0.0.1",
                       help="Proxy host (default: 127.0.0.1)")
    parser.add_argument("--proxy-port", 
                       type=int, 
                       default=9050,
                       help="Proxy port (default: 9050)")
    
    args = parser.parse_args()
    
    logger.info("Anna's Archive Connectivity Test")
    logger.info("=" * 40)
    logger.info(f"Proxy: {args.proxy_type} ({args.proxy_host}:{args.proxy_port})")
    logger.info("")
    
    exit_code = test_known_book(args.proxy_type, args.proxy_host, args.proxy_port)
    sys.exit(exit_code)