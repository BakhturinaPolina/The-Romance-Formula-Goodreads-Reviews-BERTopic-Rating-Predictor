#!/usr/bin/env python3
"""
Ingest Anna's Archive Elasticsearch data
Loads AAC JSONL files into Elasticsearch index
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_jsonl_file(file_path: str):
    """Load JSONL file, handling both .jsonl and .jsonl.zst"""
    import zstandard as zstd
    
    if file_path.endswith('.zst'):
        logger.info(f"Loading compressed file: {file_path}")
        with open(file_path, 'rb') as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                for line in reader:
                    line = line.decode('utf-8').strip()
                    if line:
                        yield json.loads(line)
    else:
        logger.info(f"Loading JSONL file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

def bulk_index_documents(es_client, index_name: str, documents, batch_size: int = 1000):
    """Bulk index documents into Elasticsearch"""
    
    def doc_generator():
        for i, doc in enumerate(documents):
            # Add document ID if not present
            doc_id = doc.get('_id') or doc.get('md5') or f"doc_{i}"
            
            yield {
                "_index": index_name,
                "_id": doc_id,
                "_source": doc
            }
    
    try:
        success_count, failed_items = bulk(
            es_client,
            doc_generator(),
            chunk_size=batch_size,
            request_timeout=60,
            max_retries=3
        )
        
        logger.info(f"Successfully indexed {success_count} documents")
        if failed_items:
            logger.warning(f"Failed to index {len(failed_items)} documents")
            for item in failed_items[:5]:  # Show first 5 failures
                logger.warning(f"Failed item: {item}")
        
        return success_count, len(failed_items)
        
    except Exception as e:
        logger.error(f"Bulk indexing failed: {e}")
        return 0, 0

def main():
    parser = argparse.ArgumentParser(description="Ingest Anna's Archive data into Elasticsearch")
    parser.add_argument("files", nargs="+", help="JSONL files to ingest")
    parser.add_argument("--index", default="aa_records", help="Elasticsearch index name")
    parser.add_argument("--host", default="http://elasticsearch:9200", help="Elasticsearch host")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for bulk indexing")
    parser.add_argument("--create-index", action="store_true", help="Create index with mapping")
    
    args = parser.parse_args()
    
    # Initialize Elasticsearch client
    es = Elasticsearch([args.host])
    
    # Test connection
    try:
        info = es.info()
        logger.info(f"Connected to Elasticsearch: {info['version']['number']}")
    except Exception as e:
        logger.error(f"Failed to connect to Elasticsearch: {e}")
        return 1
    
    # Create index if requested
    if args.create_index:
        logger.info(f"Creating index: {args.index}")
        try:
            # Basic mapping for Anna's Archive documents
            mapping = {
                "mappings": {
                    "properties": {
                        "md5": {"type": "keyword"},
                        "title": {
                            "type": "text",
                            "analyzer": "standard",
                            "fields": {
                                "keyword": {"type": "keyword"}
                            }
                        },
                        "author": {
                            "type": "text",
                            "analyzer": "standard",
                            "fields": {
                                "keyword": {"type": "keyword"}
                            }
                        },
                        "year": {"type": "integer"},
                        "language": {"type": "keyword"},
                        "extension": {"type": "keyword"},
                        "filesize": {"type": "long"},
                        "isbn10": {"type": "keyword"},
                        "isbn13": {"type": "keyword"},
                        "publisher": {"type": "text"},
                        "series": {"type": "text"}
                    }
                }
            }
            
            if es.indices.exists(index=args.index):
                logger.info(f"Index {args.index} already exists")
            else:
                es.indices.create(index=args.index, body=mapping)
                logger.info(f"Created index: {args.index}")
                
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return 1
    
    # Process files
    total_docs = 0
    total_failed = 0
    
    for file_path in args.files:
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            continue
        
        logger.info(f"Processing file: {file_path}")
        
        try:
            documents = load_jsonl_file(file_path)
            success_count, failed_count = bulk_index_documents(
                es, args.index, documents, args.batch_size
            )
            
            total_docs += success_count
            total_failed += failed_count
            
        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            continue
    
    logger.info(f"Ingestion completed!")
    logger.info(f"Total documents indexed: {total_docs}")
    logger.info(f"Total failed: {total_failed}")
    
    # Show index stats
    try:
        stats = es.indices.stats(index=args.index)
        doc_count = stats['indices'][args.index]['total']['docs']['count']
        logger.info(f"Index {args.index} now contains {doc_count} documents")
    except Exception as e:
        logger.warning(f"Could not get index stats: {e}")
    
    return 0

if __name__ == "__main__":
    exit(main())
