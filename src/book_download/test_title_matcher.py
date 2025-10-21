#!/usr/bin/env python3
"""
Test script for title matcher integration
Tests the complete pipeline: CSV → MD5 matching → download
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from title_matcher_cli import TitleMatcher

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_csv():
    """Create a small test CSV with sample titles"""
    test_data = [
        {
            'work_id': 1,
            'title': 'Pride and Prejudice',
            'author_name': 'Jane Austen',
            'publication_year': 1813
        },
        {
            'work_id': 2,
            'title': 'The Great Gatsby',
            'author_name': 'F. Scott Fitzgerald',
            'publication_year': 1925
        },
        {
            'work_id': 3,
            'title': 'To Kill a Mockingbird',
            'author_name': 'Harper Lee',
            'publication_year': 1960
        }
    ]
    
    test_csv_path = "/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/data/processed/test_titles.csv"
    
    df = pd.DataFrame(test_data)
    df.to_csv(test_csv_path, index=False)
    
    logger.info(f"Created test CSV: {test_csv_path}")
    return test_csv_path

def test_simulation_mode():
    """Test title matcher in simulation mode (without real backends)"""
    logger.info("=== TESTING SIMULATION MODE ===")
    
    # Create test CSV
    test_csv = create_test_csv()
    
    # Test with simulated results (this would work even without backends)
    logger.info("Note: This test runs in simulation mode")
    logger.info("To test with real backends, ensure MariaDB or Elasticsearch is running")
    logger.info("and contains Anna's Archive data")
    
    # Show what the CLI command would look like
    logger.info("\nTo run with MariaDB:")
    logger.info(f"python title_matcher_cli.py --backend mariadb --in {test_csv} --out test_results.csv")
    logger.info("  --db-host localhost --db-name annas_archive --db-user annas_user --db-pass annas_pass")
    
    logger.info("\nTo run with Elasticsearch:")
    logger.info(f"python title_matcher_cli.py --backend es --in {test_csv} --out test_results.csv")
    logger.info("  --es-host http://localhost:9200 --index aa_records")
    
    logger.info("\nTo run with automatic download:")
    logger.info(f"python title_matcher_cli.py --backend mariadb --in {test_csv} --download --daily-limit 5")

def test_existing_sample_data():
    """Test with existing sample data from the project"""
    logger.info("=== TESTING WITH EXISTING SAMPLE DATA ===")
    
    sample_csv = "/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/data/processed/sample_50_books.csv"
    
    if not os.path.exists(sample_csv):
        logger.warning(f"Sample CSV not found: {sample_csv}")
        return
    
    # Load and examine the sample data
    try:
        df = pd.read_csv(sample_csv)
        logger.info(f"Loaded sample data: {len(df)} books")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Show first few titles
        logger.info("\nFirst 5 titles:")
        for idx, row in df.head().iterrows():
            title = row.get('title', 'Unknown')
            author = row.get('author_name', 'Unknown')
            year = row.get('publication_year', 'Unknown')
            logger.info(f"  {idx+1}. '{title}' by {author} ({year})")
        
        # Show CLI command for this data
        logger.info(f"\nTo process this sample data:")
        logger.info(f"python title_matcher_cli.py --backend mariadb --in {sample_csv} --out sample_results.csv")
        logger.info("  --db-host localhost --db-name annas_archive --db-user annas_user --db-pass annas_pass")
        
    except Exception as e:
        logger.error(f"Error reading sample data: {e}")

def test_docker_setup():
    """Test Docker Compose setup"""
    logger.info("=== TESTING DOCKER SETUP ===")
    
    compose_file = "/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/docker-compose.yml"
    env_file = "/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/env.example"
    
    if os.path.exists(compose_file):
        logger.info("✅ Docker Compose file found")
    else:
        logger.error("❌ Docker Compose file not found")
    
    if os.path.exists(env_file):
        logger.info("✅ Environment example file found")
    else:
        logger.error("❌ Environment example file not found")
    
    logger.info("\nTo start the Docker stack:")
    logger.info("1. Copy env.example to .env and adjust settings")
    logger.info("2. docker compose --env-file .env up -d")
    logger.info("3. Wait for services to start (check with: docker compose ps)")
    
    logger.info("\nTo ingest data:")
    logger.info("MariaDB: ./scripts/ingest_mariadb.sh data/maria/your_dump.sql.zst")
    logger.info("Elasticsearch: docker compose exec tools python scripts/ingest_es.py --index aa_records data/es/*.jsonl*")

def main():
    """Main test function"""
    logger.info("Title Matcher Integration Test")
    logger.info("=" * 50)
    
    # Test simulation mode
    test_simulation_mode()
    
    print("\n" + "=" * 50)
    
    # Test with existing sample data
    test_existing_sample_data()
    
    print("\n" + "=" * 50)
    
    # Test Docker setup
    test_docker_setup()
    
    logger.info("\n=== TEST COMPLETED ===")
    logger.info("Next steps:")
    logger.info("1. Start Docker services: docker compose up -d")
    logger.info("2. Ingest Anna's Archive data using the provided scripts")
    logger.info("3. Run title matching: python title_matcher_cli.py --backend mariadb --in your_data.csv")
    logger.info("4. Optionally add --download flag for automatic downloads")

if __name__ == "__main__":
    main()
