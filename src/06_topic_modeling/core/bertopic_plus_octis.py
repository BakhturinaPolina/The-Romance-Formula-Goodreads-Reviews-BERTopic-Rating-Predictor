# -*- coding: utf-8 -*-
"""
BERTopic + OCTIS pipeline for reviews corpus topic modeling.

Adapted from original novels pipeline to work with Goodreads reviews.
Key changes:
- Loads raw (unpreprocessed) sentences from reviews
- Uses work_id for book-level topic analysis
- Preserves pop_tier for correlation analysis
- GPU/CPU fallback support
- Uses all-mpnet-base-v2 embedding model
"""

import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths (go up 4 levels: file -> core -> 06_topic_modeling -> src -> project_root)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_INTERIM = PROJECT_ROOT / "data" / "intermediate"

# Default paths (can be overridden)
dataset_path = DATA_PROCESSED / "review_sentences_for_bertopic.parquet"
octis_dataset_path = DATA_INTERIM / "octis_reviews"

import subprocess
import sys
import importlib
import time
import os
import pickle
import traceback
import gensim.corpora as corpora


# List of required libraries with their import names if different
required_libraries = {
    "bertopic": "bertopic",
    "octis": "octis",
    "sentence-transformers": "sentence_transformers",
    "umap-learn": "umap",
    "hdbscan": "hdbscan",
    "tqdm": "tqdm",
    "pandas": "pandas",
    "gensim": "gensim",
    "scipy": "scipy",
    "pyyaml": "yaml"
}


# Function to install a library
def install_and_import(package, import_name=None):
    if import_name is None:
        import_name = package
    try:
        importlib.import_module(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    finally:
        globals()[import_name] = importlib.import_module(import_name)


# Install and import required libraries
for package, import_name in required_libraries.items():
    install_and_import(package, import_name)

# Import libs

import pandas as pd
import re
import numpy as np
from tqdm import tqdm
import csv
import json
import pprint

from skopt.space.space import Real, Categorical, Integer

from scipy.sparse import csr_matrix

# GPU/CPU fallback for UMAP and HDBSCAN
# Fix cuML library path issue - cuML libraries are in libcuml/lib64 but not in system path
import os
import sys
try:
    # Try to find and add cuML library path to LD_LIBRARY_PATH
    site_packages = None
    for p in sys.path:
        if 'venv' in p and 'site-packages' in p:
            site_packages = p
            break
    if site_packages:
        libcuml_path = os.path.join(site_packages, 'libcuml', 'lib64')
        if os.path.exists(libcuml_path):
            # Add to LD_LIBRARY_PATH environment variable
            current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
            if libcuml_path not in current_ld_path:
                os.environ['LD_LIBRARY_PATH'] = f"{libcuml_path}:{current_ld_path}" if current_ld_path else libcuml_path
                logger.debug(f"Added cuML library path to LD_LIBRARY_PATH: {libcuml_path}")
except Exception as e:
    logger.debug(f"Could not set cuML library path: {e}")

try:
    from cuml.cluster import HDBSCAN as GPU_HDBSCAN
    from cuml.manifold import UMAP as GPU_UMAP
    USE_GPU = True
    logger.info("✓ GPU libraries (cuML) available - will use GPU for UMAP/HDBSCAN")
except ImportError as e:
    from hdbscan import HDBSCAN as CPU_HDBSCAN
    from umap import UMAP as CPU_UMAP
    USE_GPU = False
    logger.warning(f"⚠ GPU libraries (cuML) not available - falling back to CPU for UMAP/HDBSCAN")
    logger.debug(f"   Import error: {e}")
    GPU_HDBSCAN = CPU_HDBSCAN
    GPU_UMAP = CPU_UMAP
except Exception as e:
    # Handle other errors like library loading issues
    from hdbscan import HDBSCAN as CPU_HDBSCAN
    from umap import UMAP as CPU_UMAP
    USE_GPU = False
    logger.warning(f"⚠ GPU libraries (cuML) encountered error - falling back to CPU for UMAP/HDBSCAN")
    logger.debug(f"   Error: {e}")
    GPU_HDBSCAN = CPU_HDBSCAN
    GPU_UMAP = CPU_UMAP

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech
from bertopic.vectorizers import ClassTfidfTransformer

# Use local optimizer class
try:
    from .optimizer import Optimizer
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from optimizer import Optimizer
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.dataset.dataset import Dataset
from octis.models.model import AbstractModel

import pandas as pd
import csv
import re

import gc
import torch
import yaml
import argparse
import ast

# Email notifications removed - use logging instead

# Parse command-line arguments for optimization
def parse_args():
    """Parse command-line arguments for memory optimization."""
    parser = argparse.ArgumentParser(
        description='BERTopic + OCTIS Optimization with memory-efficient options'
    )
    parser.add_argument(
        '--max-embedding-models',
        type=int,
        default=None,
        help='Maximum number of embedding models to process (default: all from config)'
    )
    parser.add_argument(
        '--embedding-batch-size',
        type=int,
        default=16,
        help='Batch size for embedding calculation (default: 16, lower = less memory)'
    )
    parser.add_argument(
        '--optimization-multiplier',
        type=int,
        default=15,
        help='Multiplier for optimization runs: N × parameters (default: 15, lower = faster but less thorough)'
    )
    parser.add_argument(
        '--max-sentences',
        type=int,
        default=None,
        help='Maximum number of sentences to load (for testing, default: None = all)'
    )
    parser.add_argument(
        '--process-one-model',
        action='store_true',
        help='Process one embedding model at a time (clears memory between models)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (default: config_bertopic_reviews.yaml in script directory)'
    )
    parser.add_argument(
        '--max-optimization-runs',
        type=int,
        default=None,
        help='Maximum number of optimization runs (overrides multiplier calculation, for testing)'
    )
    return parser.parse_args()


def load_config(config_path=None):
    """
    Load configuration from YAML file with backward compatibility.
    
    Args:
        config_path: Path to config file. If None, tries default location.
        
    Returns:
        dict: Configuration dictionary, or None if file not found (backward compatibility)
    """
    if config_path is None:
        # Config is now in config/ directory (one level up from core/)
        config_path = Path(__file__).parent.parent / "config" / "config_bertopic_reviews.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        logger.info("Using default configuration (backward compatibility mode)")
        return None
    
    logger.info(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("✓ Configuration loaded successfully")
        return config
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        logger.info("Falling back to default configuration (backward compatibility mode)")
        return None


def build_search_space_from_config(config):
    """
    Build skopt search space from configuration dictionary.
    
    Args:
        config: Configuration dictionary from YAML
        
    Returns:
        dict: Search space for skopt optimizer
    """
    logger.info("Building search space from configuration...")
    search_space = {}
    
    # UMAP parameters
    if 'umap_n_neighbors' in config:
        param = config['umap_n_neighbors']
        if param['type'] == 'integer':
            search_space['umap__n_neighbors'] = Integer(param['min'], param['max'])
            logger.info(f"  umap__n_neighbors: Integer({param['min']}, {param['max']})")
    
    if 'umap_n_components' in config:
        param = config['umap_n_components']
        if param['type'] == 'integer':
            search_space['umap__n_components'] = Integer(param['min'], param['max'])
            logger.info(f"  umap__n_components: Integer({param['min']}, {param['max']})")
    
    if 'umap_metric' in config:
        param = config['umap_metric']
        if param['type'] == 'categorical':
            search_space['umap__metric'] = Categorical(param['values'])
            logger.info(f"  umap__metric: Categorical({len(param['values'])} values)")
    
    # HDBSCAN parameters
    if 'hdbscan_min_cluster_size' in config:
        param = config['hdbscan_min_cluster_size']
        if param['type'] == 'integer':
            search_space['hdbscan__min_cluster_size'] = Integer(param['min'], param['max'])
            logger.info(f"  hdbscan__min_cluster_size: Integer({param['min']}, {param['max']})")
    
    if 'hdbscan_min_samples' in config:
        param = config['hdbscan_min_samples']
        if param['type'] == 'integer':
            # Note: constraint will be enforced in train_model
            search_space['hdbscan__min_samples'] = Integer(param['min'], param['max'])
            logger.info(f"  hdbscan__min_samples: Integer({param['min']}, {param['max']})")
    
    if 'hdbscan_metric' in config:
        param = config['hdbscan_metric']
        if param['type'] == 'categorical':
            search_space['hdbscan__metric'] = Categorical(param['values'])
            logger.info(f"  hdbscan__metric: Categorical({len(param['values'])} values)")
    
    # BERTopic parameters
    if 'top_n_words' in config:
        param = config['top_n_words']
        if param['type'] == 'integer':
            search_space['bertopic__top_n_words'] = Integer(param['min'], param['max'])
            logger.info(f"  bertopic__top_n_words: Integer({param['min']}, {param['max']})")
    
    if 'n_gram_range' in config:
        param = config['n_gram_range']
        if param['type'] == 'categorical':
            # Convert list values to string representations for skopt (tuples are not hashable)
            # We'll convert back to tuples in set_hyperparameters
            string_values = []
            for v in param['values']:
                if isinstance(v, list):
                    # Convert list to tuple string representation
                    tuple_val = tuple(int(x) for x in v)
                    string_val = str(tuple_val)
                elif isinstance(v, tuple):
                    # Already a tuple, ensure Python ints
                    tuple_val = tuple(int(x) for x in v)
                    string_val = str(tuple_val)
                else:
                    string_val = str(v)
                string_values.append(string_val)
            search_space['bertopic__n_gram_range'] = Categorical(string_values)
            logger.info(f"  bertopic__n_gram_range: Categorical({len(string_values)} values) - stored as strings, will convert to tuples")
    
    # NOTE: min_topic_size removed - it is ignored when custom hdbscan_model is passed to BERTopic
    # Only hdbscan__min_cluster_size controls topic size in this setup
    
    # NOTE: nr_topics removed from optimization - it can only REDUCE topics, not create them
    # Set nr_topics=None in default hyperparameters to keep all topics that HDBSCAN finds
    
    if 'low_memory' in config:
        param = config['low_memory']
        if param['type'] == 'categorical':
            # Convert string booleans to actual booleans
            bool_values = [v if isinstance(v, bool) else v.lower() == 'true' for v in param['values']]
            search_space['bertopic__low_memory'] = Categorical(bool_values)
            logger.info(f"  bertopic__low_memory: Categorical({len(bool_values)} values)")
    
    # Vectorizer parameters (keep defaults for now, can be added to config later)
    search_space['vectorizer__min_df'] = Real(0.001, 0.01)
    logger.info(f"  vectorizer__min_df: Real(0.001, 0.01) [default]")
    
    logger.info(f"✓ Search space built with {len(search_space)} parameters")
    return search_space


def get_default_search_space():
    """
    Get default search space for backward compatibility.
    
    Returns:
        dict: Default search space
    """
    logger.info("Using default search space (backward compatibility)")
    return {
        'umap__n_neighbors': Integer(2, 50),
        'umap__n_components': Integer(2, 10),
        'umap__min_dist': Real(0.0, 0.1),
        'hdbscan__min_cluster_size': Integer(50, 500),
        'hdbscan__min_samples': Integer(10, 100),
        'vectorizer__min_df': Real(0.001, 0.01),
        'bertopic__top_n_words': Integer(10, 40),
        # NOTE: bertopic__min_topic_size removed - it is ignored when custom hdbscan_model is used
    }




with torch.no_grad():
    try:
        # Parse command-line arguments
        args = parse_args()
        
        logger.info("=" * 80)
        logger.info("BERTopic + OCTIS Optimization for Reviews Corpus")
        logger.info("=" * 80)
        logger.info("\nMemory Optimization Settings:")
        if args.max_embedding_models:
            logger.info(f"  Max embedding models: {args.max_embedding_models}")
        logger.info(f"  Embedding batch size: {args.embedding_batch_size}")
        logger.info(f"  Optimization multiplier: {args.optimization_multiplier}")
        if args.max_sentences:
            logger.info(f"  Max sentences: {args.max_sentences:,}")
        logger.info(f"  Process one model at a time: {args.process_one_model}")
        
        # Load configuration
        logger.info("\n[STEP 1] Loading configuration...")
        config = load_config(args.config)
        if config:
            logger.info(f"  Configuration loaded: {len(config)} parameter groups")
            logger.info(f"  Available parameters: {', '.join(config.keys())}")
        else:
            logger.info("  Using default configuration (backward compatibility)")
        
        # Load raw (unpreprocessed) sentences from reviews
        logger.info("\n[STEP 2] Loading dataset...")
        # IMPORTANT: Use raw text for embeddings (sentence transformers need unpreprocessed text)
        # 
        # Option 1: Load from test parquet file (if exists, for quick testing)
        # Option 2: Load raw sentences from source reviews (for production)
        test_parquet = DATA_INTERIM / "review_sentences_test_10k.parquet"
        logger.info(f"  Checking for test dataset: {test_parquet}")
        
        if test_parquet.exists():
            logger.info(f"Found test dataset: {test_parquet}")
            logger.info("  Loading from test parquet (for quick testing)")
            logger.warning("  NOTE: Test dataset may have cleaned text. For production, use raw text from reviews.")
            try:
                from .load_raw_sentences import load_raw_sentences_from_parquet
            except ImportError:
                import sys
                sys.path.insert(0, str(Path(__file__).parent))
                from load_raw_sentences import load_raw_sentences_from_parquet
            
            df = load_raw_sentences_from_parquet(test_parquet)
        else:
            logger.info("Loading raw sentences from reviews (production mode)...")
            try:
                from .load_raw_sentences import load_raw_sentences_from_reviews
            except ImportError:
                import sys
                sys.path.insert(0, str(Path(__file__).parent))
                from load_raw_sentences import load_raw_sentences_from_reviews
            
            # Load raw sentences (unpreprocessed)
            # For testing, can use max_sentences parameter
            max_sentences_to_load = args.max_sentences if args.max_sentences else None
            logger.info(f"  Loading with max_sentences={max_sentences_to_load}")
            df = load_raw_sentences_from_reviews(
                max_sentences=max_sentences_to_load,  # Use command-line argument
                seed=42
            )
        
        logger.info(f"  ✓ Loaded {len(df):,} raw sentences")
        logger.info(f"  Columns: {list(df.columns)}")
        
        # Show distribution
        if 'pop_tier' in df.columns:
            logger.info("\n  Pop tier distribution:")
            tier_counts = df['pop_tier'].value_counts()
            for tier, count in tier_counts.items():
                pct = count / len(df) * 100
                logger.info(f"    {tier}: {count:,} ({pct:.1f}%)")
        
        # Extract sentences as list (RAW, UNPREPROCESSED - no cleaning!)
        # This is critical: sentence transformers need raw text
        logger.info("\n[STEP 3] Preparing dataset for embeddings and coherence...")
        dataset_as_list_of_strings = df['sentence_text'].tolist()
        logger.info(f"  Total sentences for embeddings: {len(dataset_as_list_of_strings):,}")
        
        # Store metadata for later use
        metadata_df = df[['sentence_id', 'review_id', 'work_id', 'pop_tier', 'rating']].copy()
        logger.info(f"  Metadata columns stored: {list(metadata_df.columns)}")
        
        # Convert sentences to list of lists for coherence calculation
        # Note: This is for coherence metrics only, embeddings use raw text
        dataset_as_list_of_lists = [sentence.split() for sentence in dataset_as_list_of_strings]
        logger.info(f"  Prepared {len(dataset_as_list_of_lists):,} sentences for coherence calculation")

        # Prepare the dataset for OCTIS
        # https://github.com/MIND-Lab/OCTIS?tab=readme-ov-file#load-a-custom-dataset
        
        logger.info("\n[STEP 4] Preparing OCTIS dataset format...")
        octis_dataset_path.mkdir(parents=True, exist_ok=True)
        octis_corpus_path = octis_dataset_path / 'corpus.tsv'
        logger.info(f"  OCTIS dataset path: {octis_dataset_path}")
        
        # Create OCTIS TSV format with work_id as label (book-level topic analysis)
        # Also preserve pop_tier in metadata for correlation analysis
        logger.info("  Creating OCTIS corpus.tsv file...")
        tsv_data = []
        for _, row in df.iterrows():
            sentence = row['sentence_text']  # Raw, unpreprocessed text
            partition = 'train'
            # Use work_id as label for book-level topic analysis
            label = str(row['work_id'])
            tsv_data.append([sentence, partition, label])
        
        # Write the data to a TSV file
        logger.info(f"  Writing {len(tsv_data):,} rows to TSV...")
        with open(octis_corpus_path, mode='w', newline='', encoding='utf-8') as tsv_file:
            tsv_writer = csv.writer(tsv_file, delimiter='\t')
            for row in tsv_data:
                tsv_writer.writerow(row)
        
        logger.info(f"  ✓ Created OCTIS corpus: {octis_corpus_path}")
        logger.info(f"  Total sentences: {len(tsv_data):,}")
        logger.info(f"  Label format: work_id (book-level)")

        logger.info("  Loading OCTIS dataset...")
        octis_dataset = Dataset()
        octis_dataset.load_custom_dataset_from_folder(str(octis_dataset_path))
        logger.info(f"  ✓ OCTIS dataset loaded successfully")

        from bertopic.representation import KeyBERTInspired, PartOfSpeech, MaximalMarginalRelevance

        # Additional Representations
        keybert_model = KeyBERTInspired()

        # Part-of-Speech
        pos_model = PartOfSpeech("en_core_web_sm")

        # MMR
        mmr_model = MaximalMarginalRelevance(diversity=0.3)

        # All representation models
        representation_model = {
            "KeyBERT": keybert_model,
            "MMR": mmr_model,
            "POS": pos_model
        }

        # Implementing custom Model for OCTIS
        # https://github.com/MIND-Lab/OCTIS?tab=readme-ov-file#implement-your-own-model

        trainings_count = 0
        training_errors = []

        class BERTopicOctisModelWithEmbeddings(AbstractModel):
            def __init__(self, embedding_model, embedding_model_name, embeddings, dataset_as_list_of_strings):
                super().__init__()

                self.embedding_model = embedding_model
                self.embedding_model_name = embedding_model_name
                self.embeddings = embeddings

                self.use_partitions = False

                # Default parameters
                self.hyperparameters = {
                    'umap': {
                        'n_neighbors': 11,
                        'n_components': 5,
                        'min_dist': 0.05,
                        'metric': 'cosine',
                        'random_state': 42
                    },
                    'hdbscan': {
                        'min_cluster_size': 150,
                        'metric': 'euclidean',
                        'cluster_selection_method': 'eom',
                        'prediction_data': True,
                        'gen_min_span_tree': True,
                        'min_samples': 20
                    },
                    'vectorizer': {
                        'stop_words': 'english',
                        'min_df': 0.005,
                        'ngram_range': (1, 1)
                    },
                    'tfdf_vectorizer': {
                        'reduce_frequent_words': True,
                        'bm25_weighting': True
                    },
                    'bertopic': {
                        'language': "english",
                        'top_n_words': 30,
                        'n_gram_range': (1, 1),
                        'min_topic_size': 127,
                        'nr_topics': None,
                        'low_memory': False,
                        'calculate_probabilities': True,
                        'verbose': True
                    }
                }

            # Override OCTIS AbstractModel train_model method
            def train_model(self, dataset, hyperparameters={}, top_words=10):
                global trainings_count
                trainings_count += 1

                logger.info(f"\n{'='*80}")
                logger.info(f"Training #{trainings_count}")
                logger.info(f"Embedding model: {self.embedding_model_name}")
                logger.info(f"{'='*80}")
                
                results_dir = DATA_INTERIM / 'octis_reviews' / 'optimization_results' / self.embedding_model_name
                results_dir.mkdir(parents=True, exist_ok=True)
                file_path = results_dir / 'result.json'
                results_csv_path = results_dir / 'results.csv'
                logger.info(f"Results directory: {results_dir}")
                logger.info(f"Results file: {file_path}")
                logger.info(f"Results CSV: {results_csv_path}")
                
                # Helper function to append results to CSV
                def append_to_results_csv(iteration, hyperparams, n_topics, accepted, error=None):
                    """Append training result to CSV file with number of topics and hyperparameters."""
                    file_exists = results_csv_path.exists()
                    with open(results_csv_path, 'a', newline='') as csvfile:
                        fieldnames = ['iteration', 'training_run', 'n_topics', 'accepted', 
                                     'umap__n_neighbors', 'umap__n_components', 'umap__min_dist',
                                     'hdbscan__min_cluster_size', 'hdbscan__min_samples',
                                     'vectorizer__min_df', 'bertopic__top_n_words', 
                                     'bertopic__min_topic_size', 'error']
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        
                        if not file_exists:
                            writer.writeheader()
                        
                        row = {
                            'iteration': iteration,
                            'training_run': trainings_count,
                            'n_topics': n_topics if n_topics is not None else '',
                            'accepted': 'Yes' if accepted else 'No',
                            'umap__n_neighbors': hyperparams.get('umap', {}).get('n_neighbors', ''),
                            'umap__n_components': hyperparams.get('umap', {}).get('n_components', ''),
                            'umap__min_dist': hyperparams.get('umap', {}).get('min_dist', ''),
                            'hdbscan__min_cluster_size': hyperparams.get('hdbscan', {}).get('min_cluster_size', ''),
                            'hdbscan__min_samples': hyperparams.get('hdbscan', {}).get('min_samples', ''),
                            'vectorizer__min_df': hyperparams.get('vectorizer', {}).get('min_df', ''),
                            'bertopic__top_n_words': hyperparams.get('bertopic', {}).get('top_n_words', ''),
                            'bertopic__min_topic_size': hyperparams.get('bertopic', {}).get('min_topic_size', ''),
                            'error': str(error) if error else ''
                        }
                        writer.writerow(row)

                if os.path.exists(file_path):
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                        result_current_call = data.get('current_call', 0)
                        # print("Result current call:",result_current_call)
                        if result_current_call > 1 and trainings_count == 1:
                            print("!!! Skipping training run to avoid potential error")
                            output_dict = {'topics': None}
                            return output_dict


                logger.info("\n[Training] Setting hyperparameters...")
                logger.info(f"  Received {len(hyperparameters)} hyperparameters from optimizer")
                self.set_hyperparameters(hyperparameters)
                logger.info("  Hyperparameters after merging:")
                for section, params in self.hyperparameters.items():
                    logger.info(f"    {section}:")
                    for key, value in params.items():
                        logger.info(f"      {key}: {value}")
                
                # Enforce constraint: hdbscan_min_samples ≤ hdbscan_min_cluster_size
                hdbscan_min_cluster_size = self.hyperparameters['hdbscan'].get('min_cluster_size', 150)
                hdbscan_min_samples = self.hyperparameters['hdbscan'].get('min_samples', 20)
                if hdbscan_min_samples > hdbscan_min_cluster_size:
                    logger.warning(f"  ⚠ Constraint violation: min_samples ({hdbscan_min_samples}) > min_cluster_size ({hdbscan_min_cluster_size})")
                    logger.info(f"  Enforcing constraint: setting min_samples = min_cluster_size = {hdbscan_min_cluster_size}")
                    self.hyperparameters['hdbscan']['min_samples'] = hdbscan_min_cluster_size
                    logger.info(f"  ✓ Constraint enforced")
                
                # Validate HDBSCAN metric compatibility with GPU/CPU version
                hdbscan_metric = self.hyperparameters['hdbscan'].get('metric', 'euclidean')
                CUML_SUPPORTED_METRICS = ['euclidean', 'l2']
                if USE_GPU and hdbscan_metric not in CUML_SUPPORTED_METRICS:
                    error_msg = (
                        f"HDBSCAN metric '{hdbscan_metric}' is not supported by cuML (GPU) version. "
                        f"cuML HDBSCAN only supports: {CUML_SUPPORTED_METRICS}. "
                        f"Please update config to use 'euclidean' or 'l2' for hdbscan_metric."
                    )
                    logger.error(f"  ✗ {error_msg}")
                    raise ValueError(error_msg)
                
                print("Training with parameters:")
                pprint.pprint(self.hyperparameters)

                # Preventing Stochastic Behavior
                # https://colab.research.google.com/drive/1BoQ_vakEVtojsd2x_U6-_x52OOuqruj2#scrollTo=28_EVoOfyZLb
                logger.info("\n[Training] Initializing pipeline components...")
                logger.info("  Creating UMAP model...")
                umap_params = self.hyperparameters['umap']
                logger.info(f"    Parameters: {umap_params}")
                # Use GPU or CPU version based on availability
                if USE_GPU:
                    logger.info("    Using GPU version (cuML)")
                    umap_model = GPU_UMAP(**umap_params)
                else:
                    logger.info("    Using CPU version")
                    umap_model = CPU_UMAP(**umap_params)
                logger.info("    ✓ UMAP model created")

                # Controlling Number of Topics
                # https://colab.research.google.com/drive/1BoQ_vakEVtojsd2x_U6-_x52OOuqruj2#scrollTo=TH6vZPGU2zpg
                logger.info("  Creating HDBSCAN model...")
                hdbscan_params = self.hyperparameters['hdbscan']
                logger.info(f"    Parameters: {hdbscan_params}")
                # Use GPU or CPU version based on availability
                if USE_GPU:
                    logger.info("    Using GPU version (cuML)")
                    hdbscan_model = GPU_HDBSCAN(**hdbscan_params)
                else:
                    logger.info("    Using CPU version")
                    hdbscan_model = CPU_HDBSCAN(**hdbscan_params)
                logger.info("    ✓ HDBSCAN model created")

                # Improving Default Representation
                # https://colab.research.google.com/drive/1BoQ_vakEVtojsd2x_U6-_x52OOuqruj2#scrollTo=66zgeCyf0jy3&line=1&uniqifier=1
                logger.info("  Creating CountVectorizer...")
                vectorizer_params = self.hyperparameters['vectorizer']
                logger.info(f"    Parameters: {vectorizer_params}")
                vectorizer_model = CountVectorizer(**vectorizer_params)
                logger.info("    ✓ CountVectorizer created")

                # Using ClassTfidfTransformer instead of CountVectorizer
                logger.info("  Creating ClassTfidfTransformer...")
                tfdf_params = self.hyperparameters['tfdf_vectorizer']
                logger.info(f"    Parameters: {tfdf_params}")
                tfdf_model = ClassTfidfTransformer(**tfdf_params)
                logger.info("    ✓ ClassTfidfTransformer created")

                logger.info("  Creating BERTopic model...")
                bertopic_params = self.hyperparameters['bertopic']
                logger.info(f"    BERTopic parameters: {bertopic_params}")
                topic_model = BERTopic(
                    # Pipeline models
                    embedding_model=self.embedding_model,
                    umap_model=umap_model,
                    hdbscan_model=hdbscan_model,
                    vectorizer_model=vectorizer_model,
                    representation_model=representation_model,
                    # Hyperparameters
                    **bertopic_params
                )
                logger.info("    ✓ BERTopic model created")

                logger.info("\n[Training] Fitting BERTopic model...")
                logger.info(f"  Dataset size: {len(dataset_as_list_of_strings):,} sentences")
                logger.info(f"  Embeddings shape: {self.embeddings.shape}")
                logger.info(f"  Starting fit_transform...")
                start_fit_time = time.time()

                try:
                    topics, probabilities = topic_model.fit_transform(dataset_as_list_of_strings, embeddings=self.embeddings)
                    fit_elapsed = time.time() - start_fit_time
                    logger.info(f"  ✓ Model fitted in {fit_elapsed:.1f} seconds ({fit_elapsed/60:.2f} minutes)")

                    # Calculate number of topics (excluding outliers with topic -1)
                    # According to BERTopic API: topics_ is a list of topic assignments
                    # https://maartengr.github.io/BERTopic/api/bertopic.html
                    logger.info("\n[Training] Analyzing topic assignments...")
                    unique_topics = set(topics)
                    n_topics = len([t for t in unique_topics if t != -1])  # Exclude outlier topic -1
                    n_outliers = list(topics).count(-1)
                    outlier_pct = (n_outliers / len(topics)) * 100 if len(topics) > 0 else 0
                    
                    logger.info(f"  Total topic assignments: {len(topics):,}")
                    logger.info(f"  Unique topics (including outliers): {len(unique_topics)}")
                    logger.info(f"  Number of topics (excluding outliers): {n_topics}")
                    logger.info(f"  Number of outliers (topic -1): {n_outliers:,} ({outlier_pct:.1f}%)")
                    
                    # NOTE: Removed MIN_TOPICS rejection check - let optimizer evaluate models based on
                    # coherence (c_v) and diversity metrics, not just topic count. Models with 10, 50, or
                    # 300 topics should all be evaluated to find the optimal configuration.

                    logger.info("\n[Training] Cleaning up GPU memory...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info("  ✓ CUDA cache cleared")
                    gc.collect()
                    logger.info("  ✓ Garbage collection completed")

                    # Creating the required output dictionary
                    logger.info("\n[Training] Extracting topic words...")
                    output_dict = {}
                    output_dict['topics'] = []
                    # Note: n_topics is not included in output_dict as it's not a standard OCTIS field
                    # and causes errors in save_model_output. It's only used for logging/CSV.

                    logger.info("  Creating Gensim dictionary for coherence calculation...")
                    dictionary = corpora.Dictionary(dataset_as_list_of_lists)
                    logger.info(f"  Dictionary size: {len(dictionary)} unique tokens")

                    # https://github.com/MaartenGr/BERTopic/issues/90#issuecomment-820915389
                    logger.info(f"  Extracting words for {n_topics} topics...")
                    topics_to_process = len(set(topics)) - topic_model._outliers
                    logger.info(f"  Processing {topics_to_process} topics (excluding outliers)")
                    
                    for topic in range(topics_to_process):
                        words = list(zip(*topic_model.get_topic(topic)))[0]
                        original_count = len(words)
                        words = [word for word in words if word in dictionary.token2id]
                        words = [word for word in words if word.lower() != "mr"]
                        words = [word for word in words if word.lower() != "ms"]
                        filtered_count = len(words)
                        if original_count != filtered_count:
                            logger.debug(f"    Topic {topic}: {original_count} → {filtered_count} words after filtering")
                        output_dict['topics'].append(words)

                    output_dict['topics'] = [words for words in output_dict['topics'] if len(words) > 0]
                    logger.info(f"  ✓ Extracted {len(output_dict['topics'])} topics with words")

                    # If the array is empty, return None, so that OCTIS doesn't process this result
                    if not output_dict['topics']:
                        logger.warning("  ⚠ No topics extracted, returning None")
                        output_dict['topics'] = None
                    else:
                        logger.info(f"  Topics contain {sum(len(t) for t in output_dict['topics'])} total words")

                    # 2. topic-word-matrix: c-TF-IDF matrix
                    logger.info("\n[Training] Creating output matrices...")
                    logger.info("  Creating topic-word-matrix (c-TF-IDF)...")
                    output_dict['topic-word-matrix'] = np.array(topic_model.c_tf_idf_)
                    logger.info(f"    Shape: {output_dict['topic-word-matrix'].shape}")

                    # 3. topic-document-matrix: probabilities matrix
                    logger.info("  Creating topic-document-matrix (probabilities)...")
                    output_dict['topic-document-matrix'] = np.array(probabilities)
                    logger.info(f"    Shape: {output_dict['topic-document-matrix'].shape}")

                    logger.info(f"\n  ✓ Model accepted: {n_topics} topics found")
                    logger.info("  Sample topics (first 3):")
                    for i, topic_words in enumerate(output_dict['topics'][:3]):
                        logger.info(f"    Topic {i}: {topic_words[:10]}...")  # Show first 10 words
                    
                    # Log to CSV
                    try:
                        iteration = getattr(self, '_current_iteration', trainings_count)
                        append_to_results_csv(iteration, self.hyperparameters, n_topics, accepted=True)
                    except Exception as e:
                        logger.warning(f"Failed to write to CSV: {e}")

                    del topic_model

                    return output_dict
                except Exception as ex:
                    print(">>>>>>>>>>>>>>>>>>>>>****TRAINING ERROR****<<<<<<<<<<<<<<<<<<<")
                    print("Hyperparameters:", hyperparameters)
                    print("Exception:", ex)
                    print("==================================================================")
                    training_errors.append({'hyperparameters': hyperparameters, 'exception': ex, 'run_count': trainings_count})
                    # Log error to CSV
                    try:
                        iteration = getattr(self, '_current_iteration', trainings_count)
                        append_to_results_csv(iteration, self.hyperparameters, None, accepted=False, error=ex)
                    except Exception as e:
                        logger.warning(f"Failed to write error to CSV: {e}")
                    output_dict = {'topics': None}
                    return output_dict

            def set_hyperparameters(self, hyperparameters):
                # Set hyperparameter from a flat dictionary where keys are in the form 'section__hyperparameter', e.g. 'umap__n_neighbors'.
                for key, value in hyperparameters.items():
                    if '__' in key:
                        section, hyperparameter = key.split('__', 1)
                        if section in self.hyperparameters and hyperparameter in self.hyperparameters[section]:
                            # Convert numpy types to Python native types first
                            if isinstance(value, np.integer):
                                value = int(value)
                            elif isinstance(value, np.floating):
                                value = float(value)
                            elif isinstance(value, np.bool_):
                                value = bool(value)
                            elif isinstance(value, np.str_):
                                value = str(value)
                            
                            # Special handling for n_gram_range: convert string back to tuple
                            if hyperparameter == 'n_gram_range' and isinstance(value, str):
                                # Convert string representation like "(1, 2)" back to tuple
                                try:
                                    value = ast.literal_eval(value)  # Safe parsing of tuple strings
                                    # Ensure it's a tuple of ints
                                    value = tuple(int(x) for x in value)
                                except Exception as e:
                                    logger.warning(f"Failed to convert n_gram_range string '{value}' to tuple: {e}, using as-is")
                            self.hyperparameters[section][hyperparameter] = value
                        else:
                            print(f"Warning: Parameter '{key}' is not recognized.")
                    else:
                        print(f"Warning: Parameter '{key}' does not match the expected format 'section__hyperparameter'.")


        # Load embedding models from config or use default
        logger.info("\n[STEP 5] Loading embedding models...")
        if config and 'embedding_model' in config:
            embedding_config = config['embedding_model']
            if embedding_config['type'] == 'categorical':
                embedding_model_names = embedding_config['values']
                # Limit number of models if specified
                if args.max_embedding_models and len(embedding_model_names) > args.max_embedding_models:
                    logger.info(f"  Limiting to {args.max_embedding_models} models (from {len(embedding_model_names)} in config)")
                    embedding_model_names = embedding_model_names[:args.max_embedding_models]
                logger.info(f"  Using {len(embedding_model_names)} embedding models from config:")
                for i, model_name in enumerate(embedding_model_names, 1):
                    logger.info(f"    {i}. {model_name}")
            else:
                logger.warning("  Invalid embedding_model config type, using default")
                embedding_model_names = ["all-mpnet-base-v2"]
        else:
            # Backward compatibility: use default model
            embedding_model_names = ["all-mpnet-base-v2"]
            logger.info(f"  Using default embedding model: {embedding_model_names[0]}")
            logger.info("  (To use multiple models, add 'embedding_model' to config file)")

        embedding_models = []
        
        # Determine device for embeddings (GPU/CPU fallback)
        logger.info("\n  Determining compute device...")
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"  ✓ Using GPU device: {torch.cuda.get_device_name(0)}")
            logger.info(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = "cpu"
            logger.warning("  ⚠ GPU not available, using CPU for embeddings (will be slower)")

        logger.info(f"\n  Loading {len(embedding_model_names)} embedding model(s)...")
        for idx, embedding_model_name in enumerate(embedding_model_names, 1):
            logger.info(f"  [{idx}/{len(embedding_model_names)}] Loading: {embedding_model_name}")
            logger.info(f"    Device: {device}")
            start_time = time.time()
            embedding_model = SentenceTransformer(embedding_model_name, device=device)
            elapsed = time.time() - start_time
            embedding_models.append(embedding_model)
            logger.info(f"    ✓ Model loaded in {elapsed:.2f} seconds")

        # Store the pre-calculated embeddings
        # https://colab.research.google.com/drive/1BoQ_vakEVtojsd2x_U6-_x52OOuqruj2#scrollTo=sVfnYtUaxyLT
        # If processing one at a time, we'll load embeddings on-demand
        precalculated_embeddings = [] if not args.process_one_model else None


        def save_embeddings(embeddings, file_path):
            """
            Save embeddings to a file using pickle.

            Args:
            embeddings (numpy.ndarray or list of numpy.ndarray): Embedding array(s) to save
            file_path (str): Path to save the embeddings
            """
            if not isinstance(embeddings, (list, np.ndarray)):
                raise ValueError("Embeddings must be a numpy array or a list of numpy arrays")

            with open(file_path, 'wb') as f:
                pickle.dump(embeddings, f)
            print(f"Embeddings saved to {file_path}")


        def load_embeddings(file_path):
            """
            Load embeddings from a file.

            Args:
            file_path (str): Path to the saved embeddings file

            Returns:
            numpy.ndarray or list of numpy.ndarray: Loaded embeddings
            """
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    embeddings = pickle.load(f)
                print(f"Embeddings loaded from {file_path}")
                return embeddings
            else:
                print(f"No saved embeddings found at {file_path}")
                return None


        # Embedding file path
        embedding_file = DATA_INTERIM / 'octis_reviews' / 'precalculated_embeddings.pkl'

        # BERTopic works by converting documents into numerical values, called embeddings.
        # This process can be very costly, especially if we want to iterate over parameters.
        # Instead, we can calculate those embeddings once and feed them to BERTopic
        # to skip calculating embeddings each time.
        logger.info("\n[STEP 6] Calculating/loading embeddings...")
        logger.info(f"  Using raw, unpreprocessed text for embeddings")
        logger.info(f"  Number of sentences: {len(dataset_as_list_of_strings):,}")
        logger.info(f"  Embedding models to process: {len(embedding_models)}")
        
        embedding_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"  Embedding storage directory: {embedding_file.parent}")
        
        # If processing one at a time, load embeddings on-demand during optimization
        # Otherwise, pre-load all embeddings
        if not args.process_one_model:
            logger.info("  Pre-loading all embeddings (standard mode)...")
            for idx, embedding_model in enumerate(embedding_models):
                embedding_model_name = embedding_model_names[idx]
                model_embedding_file = embedding_file.parent / f"embeddings_{embedding_model_name.replace('/', '_')}.pkl"
                
                logger.info(f"\n  [{idx+1}/{len(embedding_models)}] Processing: {embedding_model_name}")
                logger.info(f"    Embedding file path: {model_embedding_file}")
                
                # Check if embeddings already exist
                if model_embedding_file.exists():
                    logger.info(f"    Found existing embeddings file")
                    logger.info(f"    Loading embeddings from: {model_embedding_file}")
                    file_size_mb = model_embedding_file.stat().st_size / (1024 * 1024)
                    logger.info(f"    File size: {file_size_mb:.1f} MB")
                    start_time = time.time()
                    embeddings = load_embeddings(str(model_embedding_file))
                    elapsed = time.time() - start_time
                    logger.info(f"    ✓ Embeddings loaded in {elapsed:.2f} seconds")
                    logger.info(f"    Embedding shape: {embeddings.shape}")
                    logger.info(f"    Embedding dtype: {embeddings.dtype}")
                    precalculated_embeddings.append(embeddings)
                else:
                    logger.info(f"    No existing embeddings found, calculating new embeddings...")
                    logger.info(f"    This may take several minutes for {len(dataset_as_list_of_strings):,} sentences...")
                    logger.info(f"    Using device: {device}")
                    logger.info(f"    Batch size: {args.embedding_batch_size} (lower = less memory)")
                    start_time = time.time()
                    embeddings = embedding_model.encode(
                        dataset_as_list_of_strings,
                        show_progress_bar=True,
                        batch_size=args.embedding_batch_size,
                        convert_to_numpy=True
                    )
                    elapsed = time.time() - start_time
                    logger.info(f"    ✓ Embeddings calculated in {elapsed/60:.1f} minutes ({elapsed:.1f} seconds)")
                    logger.info(f"    Embedding shape: {embeddings.shape}")
                    logger.info(f"    Embedding dtype: {embeddings.dtype}")
                    logger.info(f"    Memory usage: {embeddings.nbytes / 1e9:.2f} GB")
                    
                    # Save embeddings
                    logger.info(f"    Saving embeddings to: {model_embedding_file}")
                    save_start = time.time()
                    save_embeddings(embeddings, str(model_embedding_file))
                    save_elapsed = time.time() - save_start
                    logger.info(f"    ✓ Embeddings saved in {save_elapsed:.2f} seconds")
                    precalculated_embeddings.append(embeddings)
        else:
            logger.info("  Embeddings will be loaded on-demand (one model at a time mode)")

        def optimize_hyperparameters(model, label, dataset_as_list_of_lists, embedding_file_path=None):
            logger.info(f"\n[STEP 7] Setting up hyperparameter optimization for: {label}")
            
            # Build search space from config or use defaults
            if config:
                logger.info("  Building search space from configuration...")
                search_space = build_search_space_from_config(config)
            else:
                logger.info("  Using default search space (backward compatibility)")
                search_space = get_default_search_space()
            
            logger.info(f"  Search space contains {len(search_space)} parameters:")
            for param_name in sorted(search_space.keys()):
                logger.info(f"    - {param_name}")

            # As a rule of thumb, if you have N hyperparameters to optimize, then you should make at least 15 times N iterations.
            # https://colab.research.google.com/github/MIND-Lab/OCTIS/blob/master/examples/OCTIS_Optimizing_CTM.ipynb#scrollTo=njjkNjl9CJW8&line=1&uniqifier=1
            if args.max_optimization_runs:
                optimization_runs = args.max_optimization_runs
                logger.info(f"  Optimization runs: {optimization_runs} (overridden by --max-optimization-runs)")
            else:
                optimization_runs = len(search_space) * args.optimization_multiplier
                logger.info(f"  Optimization runs: {optimization_runs} ({args.optimization_multiplier} × {len(search_space)} parameters)")
                if args.optimization_multiplier < 15:
                    logger.warning(f"  ⚠ Using reduced multiplier ({args.optimization_multiplier} < 15) - optimization may be less thorough")

            # Topic models are usually probabilistic and thus produce different results even with the same hyperparameter configuration.
            # So we run the model multiple times and then take the median of the evaluated metric to get a more reliable result.
            model_runs = 1  # 1 is enough
            logger.info(f"  Model runs per configuration: {model_runs}")

            topk = 10
            logger.info(f"  Top-k for metrics: {topk}")

            logger.info("  Filtering empty documents for coherence calculation...")
            original_count = len(dataset_as_list_of_lists)
            dataset_as_list_of_lists = [doc for doc in dataset_as_list_of_lists if len(doc) > 0]
            filtered_count = len(dataset_as_list_of_lists)
            if original_count != filtered_count:
                logger.info(f"    Filtered {original_count - filtered_count} empty documents")
            logger.info(f"  Documents for coherence: {filtered_count:,}")

            # Use a different coherence measure
            logger.info("  Initializing evaluation metrics...")
            logger.info(f"    Coherence metric: c_v (topk={topk})")
            npmi = Coherence(texts=dataset_as_list_of_lists, topk=topk, measure='c_v')
            logger.info(f"    Diversity metric: TopicDiversity (topk={topk})")
            diversity = TopicDiversity(topk=10)  # Initialize metric
            logger.info("  ✓ Metrics initialized")

            logger.info("  Creating optimizer instance...")
            optimizer = Optimizer()
            logger.info("  ✓ Optimizer created")

            results_path = str(DATA_INTERIM / 'octis_reviews' / 'optimization_results' / label) + '/'
            result_file = results_path + 'result.json'
            logger.info(f"  Results path: {results_path}")
            logger.info(f"  Result file: {result_file}")

            if os.path.isfile(result_file):
                logger.info(f"\n  Found existing optimization results")
                logger.info(f"  Resuming optimization from: {result_file}")
                file_size_mb = os.path.getsize(result_file) / (1024 * 1024)
                logger.info(f"  Result file size: {file_size_mb:.1f} MB")
                try:
                    optimization_result = optimizer.resume_optimization(result_file, model=model)
                    logger.info("  ✓ Optimization resumed successfully")
                    return optimization_result
                except (ValueError, json.JSONDecodeError) as e:
                    if "corrupted" in str(e).lower() or "jsondecodeerror" in str(type(e).__name__).lower():
                        logger.warning(f"  ⚠ Cannot resume from corrupted result file: {e}")
                        logger.info(f"  Starting fresh optimization instead...")
                        # Remove the corrupted file if it still exists
                        if os.path.exists(result_file):
                            try:
                                os.remove(result_file)
                                logger.info(f"  ✓ Removed corrupted result file")
                            except Exception as remove_err:
                                logger.warning(f"  Could not remove corrupted file: {remove_err}")
                        # Fall through to start fresh optimization
                    else:
                        raise  # Re-raise if it's a different error
            else:
                logger.info(f"\n  No existing results found")
                logger.info(f"  Starting new optimization")
                logger.info(f"  Results will be saved to: {results_path}")
                logger.info(f"  Optimization settings:")
                logger.info(f"    - Number of calls: {optimization_runs}")
                logger.info(f"    - Model runs per config: {model_runs}")
                logger.info(f"    - Primary metric: Coherence (c_v)")
                logger.info(f"    - Extra metrics: TopicDiversity")
                logger.info(f"    - Save models: True")
                optimization_result = optimizer.optimize(
                    model, octis_dataset, npmi, search_space, number_of_call=optimization_runs,
                    model_runs=model_runs, save_models=True,
                    extra_metrics=[diversity],  # to keep track of other metrics
                    save_path=results_path
                )
                logger.info("  ✓ Optimization completed")
                return optimization_result

        # @title
        # Original Code!
        optimization_results = []

        # for index, embedding_model in enumerate(embedding_models):
        #     embedding_model_name = embedding_model_names[index]
        #     print("Instantiating BERTopicOctisModelWithEmbeddings for ", embedding_model_name)
        #     model = BERTopicOctisModelWithEmbeddings(embedding_model=embedding_model,
        #                                              embeddings=precalculated_embeddings[index],
        #                                              dataset_as_list_of_strings=dataset_as_list_of_strings)
        #
        #     print("Optimizing hyperparameters for ", embedding_model_name)
        #     optimization_result = optimize_hyperparameters(model, embedding_model_name)
        #     optimization_results.append(optimization_result)
        #     print("=== Optimized hyperparameters for embedding_model_name:", embedding_model_name)

        optimization_errors = []
        # monitor = Monitor(10)


        def optimize_with_restart(model, embedding_model_name, dataset_as_list_of_lists, embedding_file_path=None, max_retries=3, delay=5):
            """
            Attempts to run optimize_hyperparameters and restarts if an exception occurs.

            :param model: The model to optimize
            :param embedding_model_name: Name of the embedding model
            :param dataset_as_list_of_lists: The dataset
            :param embedding_file_path: Path to embedding file (for on-demand loading)
            :param max_retries: Maximum number of retry attempts
            :param delay: Delay in seconds between retries
            :return: The optimization result or None if all attempts fail
            """
            for attempt in range(max_retries):
                try:
                    print(f"Attempt {attempt + 1} of {max_retries}")
                    optimization_result = optimize_hyperparameters(model, embedding_model_name,
                                                                   dataset_as_list_of_lists, embedding_file_path)
                    return optimization_result
                except Exception as e:
                    print(f"An error occurred during optimization (Attempt {attempt + 1}):")
                    print(traceback.format_exc())
                    # if attempt < max_retries - 1:
                    print(f"Restarting in {delay} seconds...")
                    time.sleep(delay)
                    # else:
                    #     print("Max retries reached. Optimization failed.")

            return None

        logger.info("\n[STEP 8] Starting optimization for all embedding models...")
        logger.info(f"  Total embedding models to optimize: {len(embedding_models)}")
        if args.process_one_model:
            logger.info("  Mode: Process one model at a time (memory-efficient)")
        
        for index, embedding_model in enumerate(embedding_models):
            try:
                embedding_model_name = embedding_model_names[index]
                logger.info(f"\n{'='*80}")
                logger.info(f"Processing embedding model {index+1}/{len(embedding_models)}: {embedding_model_name}")
                logger.info(f"{'='*80}")
                
                # If processing one at a time, load embeddings on-demand
                if args.process_one_model:
                    logger.info("  Loading embeddings for this model only...")
                    model_embedding_file = embedding_file.parent / f"embeddings_{embedding_model_name.replace('/', '_')}.pkl"
                    
                    if model_embedding_file.exists():
                        logger.info(f"    Loading from: {model_embedding_file}")
                        current_embeddings = load_embeddings(str(model_embedding_file))
                    else:
                        logger.info(f"    Calculating embeddings for {embedding_model_name}...")
                        logger.info(f"    This may take several minutes for {len(dataset_as_list_of_strings):,} sentences...")
                        logger.info(f"    Batch size: {args.embedding_batch_size}")
                        start_time = time.time()
                        current_embeddings = embedding_model.encode(
                            dataset_as_list_of_strings,
                            show_progress_bar=True,
                            batch_size=args.embedding_batch_size,
                            convert_to_numpy=True
                        )
                        elapsed = time.time() - start_time
                        logger.info(f"    ✓ Embeddings calculated in {elapsed/60:.1f} minutes")
                        logger.info(f"    Saving embeddings...")
                        save_embeddings(current_embeddings, str(model_embedding_file))
                        logger.info(f"    ✓ Embeddings saved")
                else:
                    current_embeddings = precalculated_embeddings[index]
                
                logger.info("  Instantiating BERTopicOctisModelWithEmbeddings...")
                model = BERTopicOctisModelWithEmbeddings(
                    embedding_model=embedding_model,
                    embedding_model_name=embedding_model_names[index],
                    embeddings=current_embeddings,
                    dataset_as_list_of_strings=dataset_as_list_of_strings
                )
                logger.info("  ✓ Model instance created")

                logger.info(f"  Starting hyperparameter optimization for {embedding_model_name}...")
                optimization_result = optimize_with_restart(model, embedding_model_name, dataset_as_list_of_lists, embedding_file)
                logger.info(f"  ✓ Optimization completed for {embedding_model_name}")
                # optimization_results.append(optimization_result)

                # Clean up
                logger.info("  Cleaning up model instance...")
                del model
                del optimization_result
                if args.process_one_model:
                    # Clear embeddings after processing
                    logger.info("  Clearing embeddings from memory...")
                    del current_embeddings
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                logger.info("  ✓ Cleanup completed")

            except Exception as ex:
                embedding_model_name = embedding_model_names[index]
                logger.error(f"\n{'='*80}")
                logger.error(f"OPTIMIZATION ERROR for model: {embedding_model_name}")
                logger.error(f"{'='*80}")
                logger.error(f"Exception type: {type(ex).__name__}")
                logger.error(f"Exception message: {str(ex)}")
                logger.error("Full traceback:", exc_info=True)
                optimization_errors.append({embedding_model_name: ex})

        # monitor.stop()

        # Inspect the optimization results
        for index, embedding_model_name in enumerate(embedding_model_names):
            file_path = DATA_INTERIM / 'octis_reviews' / 'optimization_results' / embedding_model_name / 'result.json'
            if file_path.exists():
                with open(file_path, 'r') as file:
                    res = json.load(file)
                    logger.info(f"\nOptimization results for {embedding_model_name}:")
                    pprint.pprint(res)
            else:
                logger.warning(f"Results file not found: {file_path}")

        logger.info("\n" + "=" * 80)
        logger.info("Optimization Summary")
        logger.info("=" * 80)
        logger.info(f"Training runs completed: {trainings_count}")
        logger.info(f"Training errors: {len(training_errors)}")
        if training_errors:
            logger.warning(f"Training errors details: {training_errors}")
        logger.info(f"Optimization errors: {len(optimization_errors)}")
        if optimization_errors:
            logger.warning(f"Optimization errors details: {optimization_errors}")

    except Exception as ex:
        logger.error("=" * 80)
        logger.error("FATAL ERROR in optimization pipeline")
        logger.error("=" * 80)
        logger.error(f"Exception type: {type(ex).__name__}")
        logger.error(f"Exception message: {str(ex)}")
        logger.error("Full traceback:", exc_info=True)
        raise
