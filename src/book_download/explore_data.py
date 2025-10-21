#!/usr/bin/env python3
"""
Book Download Research Component - Data Exploration
Simple script to understand CSV structure and prepare for downloads
"""

import pandas as pd
import os
import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def explore_csv_data():
    """Explore the structure of our romance novel datasets"""
    
    # Define paths
    base_path = "/home/polina/Documents/goodreads_romance_research_cursor/romance-novel-nlp-research/data/processed"
    sample_csv = os.path.join(base_path, "sample_books_for_download.csv")
    main_csv = os.path.join(base_path, "romance_subdataset_6000.csv")
    
    logger.info("=== ROMANCE NOVEL DATA EXPLORATION ===")
    logger.info(f"Base path: {base_path}")
    logger.info(f"Sample CSV: {sample_csv}")
    logger.info(f"Main CSV: {main_csv}")
    
    # Check if files exist
    for file_path, name in [(sample_csv, "Sample CSV"), (main_csv, "Main CSV")]:
        if os.path.exists(file_path):
            logger.info(f"✓ {name} exists")
        else:
            logger.error(f"✗ {name} NOT FOUND: {file_path}")
            return False
    
    logger.info("\n=== SAMPLE CSV ANALYSIS ===")
    # Load sample CSV
    try:
        sample_df = pd.read_csv(sample_csv)
        logger.info(f"Sample CSV shape: {sample_df.shape}")
        logger.info(f"Columns: {list(sample_df.columns)}")
        print("\nFirst 3 rows:")
        print(sample_df.head(3).to_string())
        logger.info(f"Data types:\n{sample_df.dtypes}")
    except Exception as e:
        logger.error(f"Error loading sample CSV: {e}")
        return False
    
    logger.info("\n=== MAIN CSV ANALYSIS ===")
    # Load main CSV (just first few rows for exploration)
    try:
        main_df = pd.read_csv(main_csv, nrows=1000)  # Load first 1000 rows for exploration
        logger.info(f"Main CSV shape (first 1000 rows): {main_df.shape}")
        logger.info(f"Columns: {list(main_df.columns)}")
        print("\nFirst 3 rows:")
        print(main_df.head(3).to_string())
        logger.info(f"Data types:\n{main_df.dtypes}")
    except Exception as e:
        logger.error(f"Error loading main CSV: {e}")
        return False
    
    logger.info("\n=== KEY COLUMNS FOR DOWNLOAD ===")
    # Focus on title and author_name columns
    logger.info("Title column analysis:")
    logger.info(f"  - Non-null titles: {main_df['title'].notna().sum()}")
    logger.info(f"  - Sample titles: {main_df['title'].head(3).tolist()}")
    
    logger.info("Author name column analysis:")
    logger.info(f"  - Non-null authors: {main_df['author_name'].notna().sum()}")
    logger.info(f"  - Sample authors: {main_df['author_name'].head(3).tolist()}")
    
    logger.info("\n=== PUBLICATION YEAR STATS ===")
    logger.info(f"Year range: {main_df['publication_year'].min()} - {main_df['publication_year'].max()}")
    logger.info("Year distribution (top 5):")
    logger.info(f"{main_df['publication_year'].value_counts().head()}")
    
    logger.info("\n=== READY FOR DOWNLOAD TESTING ===")
    logger.info("Sample books for testing:")
    for idx, row in sample_df.head(3).iterrows():
        logger.info(f"  {idx+1}. '{row['title']}' by {row['author_name']} ({row['publication_year']})")
    
    return True

if __name__ == "__main__":
    logger.info("Starting data exploration...")
    success = explore_csv_data()
    if success:
        logger.info("Data exploration completed successfully!")
    else:
        logger.error("Data exploration failed!")
        sys.exit(1)
