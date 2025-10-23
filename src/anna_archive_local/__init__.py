"""
Anna's Archive Local Data Pipeline

This module provides an offline book search system using Anna's Archive data dumps.
It converts ElasticSearch JSON dumps to Parquet format and uses DuckDB for efficient
SQL-based queries, avoiding the need for web scraping.

Key Components:
- json_to_parquet: Convert JSON dumps to Parquet format
- query_engine: DuckDB-based search interface
- api_downloader: Fast download API client
- sample_data_extractor: Create test datasets from full dumps
"""

__version__ = "1.0.0"
__author__ = "Romance Novel NLP Research Project"
