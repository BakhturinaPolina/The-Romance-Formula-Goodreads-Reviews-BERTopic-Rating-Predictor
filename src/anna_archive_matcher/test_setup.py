#!/usr/bin/env python3
"""
Test script to validate Anna Archive Book Matcher setup
"""

import sys
import logging
from pathlib import Path
import pandas as pd

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from anna_archive_matcher.core.book_matcher import BookMatcher
from anna_archive_matcher.utils.data_processor import AnnaArchiveDataProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required modules can be imported"""
    try:
        import duckdb
        import pandas as pd
        import requests
        import tqdm
        import zstandard as zstd
        logger.info("✓ All required packages imported successfully")
        return True
    except ImportError as e:
        logger.error(f"✗ Import error: {e}")
        return False


def test_data_structure():
    """Test that data directory structure exists"""
    data_dir = Path("data")
    required_dirs = [
        "elasticsearch", "elasticsearchF",
        "aac", "aacF",
        "mariadb", "mariadbF"
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = data_dir / dir_name
        if dir_path.exists():
            logger.info(f"✓ Directory exists: {dir_path}")
        else:
            logger.warning(f"✗ Directory missing: {dir_path}")
            all_exist = False
    
    return all_exist


def test_romance_dataset():
    """Test loading the romance dataset"""
    try:
        romance_csv = Path("../../data/processed/romance_books_main_final_canonicalized.csv")
        if not romance_csv.exists():
            logger.error(f"✗ Romance dataset not found: {romance_csv}")
            return False
        
        df = pd.read_csv(romance_csv)
        logger.info(f"✓ Romance dataset loaded: {len(df)} books")
        logger.info(f"  Columns: {list(df.columns)}")
        return True
    except Exception as e:
        logger.error(f"✗ Error loading romance dataset: {e}")
        return False


def test_book_matcher():
    """Test initializing the book matcher"""
    try:
        matcher = BookMatcher("data")
        logger.info("✓ Book matcher initialized successfully")
        matcher.close()
        return True
    except Exception as e:
        logger.error(f"✗ Error initializing book matcher: {e}")
        return False


def test_data_processor():
    """Test initializing the data processor"""
    try:
        processor = AnnaArchiveDataProcessor("data")
        logger.info("✓ Data processor initialized successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Error initializing data processor: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("Testing Anna Archive Book Matcher Setup")
    logger.info("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Data Structure", test_data_structure),
        ("Romance Dataset", test_romance_dataset),
        ("Book Matcher", test_book_matcher),
        ("Data Processor", test_data_processor)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\nRunning test: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        logger.info("✓ All tests passed! Setup is ready.")
        logger.info("\nNext steps:")
        logger.info("1. Download Anna Archive datasets")
        logger.info("2. Run: python run_matcher.py --process-data")
        logger.info("3. Run: python run_matcher.py --romance-csv <your_csv>")
    else:
        logger.warning("✗ Some tests failed. Please check the setup.")
        logger.info("\nTroubleshooting:")
        logger.info("1. Install missing packages: pip install -r requirements.txt")
        logger.info("2. Create data directories: python setup_datasets.py")
        logger.info("3. Check file paths and permissions")


if __name__ == "__main__":
    main()
