#!/usr/bin/env python3
"""
Custom title matcher for Anna's Archive data structure
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elasticsearch import Elasticsearch
import logging
from typing import List, Dict, Optional, Tuple
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomTitleMatcher:
    def __init__(self, es_host: str = "http://localhost:9200", index_name: str = "aa_records"):
        self.es_client = Elasticsearch([es_host])
        self.index_name = index_name
        
        # Test connection
        try:
            info = self.es_client.info()
            logger.info(f"Elasticsearch connection successful: {info['version']['number']}")
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            raise
    
    def search_elasticsearch(self, title: str, author: str = None, year: int = None) -> List[Dict]:
        """Search Elasticsearch for matching books using the correct field structure"""
        try:
            # Build search query for nested structure
            must_clauses = [
                {"match": {"metadata.record.title": {"query": title, "fuzziness": "AUTO"}}}
            ]
            
            if author:
                must_clauses.append({"match": {"metadata.record.author": {"query": author, "fuzziness": "AUTO"}}})
            
            if year:
                must_clauses.append({"term": {"metadata.record.year": str(year)}})
            
            query = {
                "query": {"bool": {"must": must_clauses}},
                "_source": ["aacid", "metadata.record.title", "metadata.record.author", "metadata.record.year", "metadata.record.isbn", "metadata.record.publisher"],
                "size": 200
            }
            
            response = self.es_client.search(index=self.index_name, body=query)
            results = [hit["_source"] for hit in response["hits"]["hits"]]
            
            logger.debug(f"Elasticsearch returned {len(results)} candidates for '{title}'")
            return results
            
        except Exception as e:
            logger.error(f"Elasticsearch search failed: {e}")
            return []
    
    def find_best_match(self, title: str, author: str = None, year: int = None) -> Tuple[Optional[Dict], float, str]:
        """Find the best matching book"""
        candidates = self.search_elasticsearch(title, author, year)
        
        if not candidates:
            return None, 0.0, "No candidates found"
        
        # For now, just return the first candidate with a basic score
        best_match = candidates[0]
        score = 85.0  # Placeholder score
        
        return best_match, score, f"Found {len(candidates)} candidates"
    
    def process_csv(self, input_csv: str, output_csv: str):
        """Process a CSV file and find matches"""
        logger.info(f"Processing CSV: {input_csv}")
        
        # Read input CSV
        df = pd.read_csv(input_csv)
        logger.info(f"Loaded {len(df)} rows from {input_csv}")
        
        results = []
        
        for idx, row in df.iterrows():
            title = row.get('title', '')
            author = row.get('author_name', '')
            year = row.get('publication_year', None)
            
            logger.info(f"Processing {idx+1}/{len(df)}: '{title}' by {author}")
            
            # Find best match
            match, score, explanation = self.find_best_match(title, author, year)
            
            # Prepare result row
            result_row = {
                'row_index': idx,
                'input_title': title,
                'input_author': author,
                'input_year': year,
                'match_score': score,
                'match_explanation': explanation,
                'matched_title': match.get('metadata', {}).get('record', {}).get('title', '') if match else '',
                'matched_author': match.get('metadata', {}).get('record', {}).get('author', '') if match else '',
                'matched_year': match.get('metadata', {}).get('record', {}).get('year', '') if match else '',
                'matched_isbn': match.get('metadata', {}).get('record', {}).get('isbn', '') if match else '',
                'matched_publisher': match.get('metadata', {}).get('record', {}).get('publisher', '') if match else '',
                'aacid': match.get('aacid', '') if match else ''
            }
            
            results.append(result_row)
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, index=False)
        logger.info(f"Results written to: {output_csv}")
        
        # Print summary
        matches = len([r for r in results if r['match_score'] > 0])
        logger.info(f"=== MATCHING SUMMARY ===")
        logger.info(f"Total processed: {len(results)}")
        logger.info(f"Matches found: {matches}")
        logger.info(f"No matches: {len(results) - matches}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Custom Title Matcher for Anna Archive')
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--output', required=True, help='Output CSV file')
    parser.add_argument('--es-host', default='http://localhost:9200', help='Elasticsearch host')
    parser.add_argument('--index', default='aa_records', help='Elasticsearch index')
    
    args = parser.parse_args()
    
    matcher = CustomTitleMatcher(es_host=args.es_host, index_name=args.index)
    matcher.process_csv(args.input, args.output)