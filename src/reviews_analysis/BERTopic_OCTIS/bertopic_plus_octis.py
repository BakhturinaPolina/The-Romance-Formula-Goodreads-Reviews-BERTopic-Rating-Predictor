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

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"

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
    "scipy": "scipy"
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
try:
    from cuml.cluster import HDBSCAN as GPU_HDBSCAN
    from cuml.manifold import UMAP as GPU_UMAP
    USE_GPU = True
    logger.info("✓ GPU libraries (cuML) available - will use GPU for UMAP/HDBSCAN")
except ImportError:
    from hdbscan import HDBSCAN as CPU_HDBSCAN
    from umap import UMAP as CPU_UMAP
    USE_GPU = False
    logger.warning("⚠ GPU libraries (cuML) not available - falling back to CPU for UMAP/HDBSCAN")
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

# Email notifications removed - use logging instead




with torch.no_grad():
    try:
        logger.info("=" * 80)
        logger.info("BERTopic + OCTIS Optimization for Reviews Corpus")
        logger.info("=" * 80)
        
        # Load raw (unpreprocessed) sentences from reviews
        # IMPORTANT: Use raw text for embeddings (sentence transformers need unpreprocessed text)
        # 
        # Option 1: Load from test parquet file (if exists, for quick testing)
        # Option 2: Load raw sentences from source reviews (for production)
        test_parquet = DATA_INTERIM / "review_sentences_test_10k.parquet"
        
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
            df = load_raw_sentences_from_reviews(
                max_sentences=None,  # Set to e.g., 10000 for testing
                seed=42
            )
        
        logger.info(f"Loaded {len(df):,} raw sentences")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Show distribution
        if 'pop_tier' in df.columns:
            logger.info("\nPop tier distribution:")
            tier_counts = df['pop_tier'].value_counts()
            for tier, count in tier_counts.items():
                pct = count / len(df) * 100
                logger.info(f"  {tier}: {count:,} ({pct:.1f}%)")
        
        # Extract sentences as list (RAW, UNPREPROCESSED - no cleaning!)
        # This is critical: sentence transformers need raw text
        dataset_as_list_of_strings = df['sentence_text'].tolist()
        logger.info(f"Total sentences for embeddings: {len(dataset_as_list_of_strings):,}")
        
        # Store metadata for later use
        metadata_df = df[['sentence_id', 'review_id', 'work_id', 'pop_tier', 'rating']].copy()
        
        # Convert sentences to list of lists for coherence calculation
        # Note: This is for coherence metrics only, embeddings use raw text
        dataset_as_list_of_lists = [sentence.split() for sentence in dataset_as_list_of_strings]
        logger.info(f"Prepared {len(dataset_as_list_of_lists):,} sentences for coherence calculation")

        # Prepare the dataset for OCTIS
        # https://github.com/MIND-Lab/OCTIS?tab=readme-ov-file#load-a-custom-dataset
        
        logger.info("\nPreparing OCTIS dataset format...")
        octis_dataset_path.mkdir(parents=True, exist_ok=True)
        octis_corpus_path = octis_dataset_path / 'corpus.tsv'
        
        # Create OCTIS TSV format with work_id as label (book-level topic analysis)
        # Also preserve pop_tier in metadata for correlation analysis
        logger.info("Creating OCTIS corpus.tsv file...")
        tsv_data = []
        for _, row in df.iterrows():
            sentence = row['sentence_text']  # Raw, unpreprocessed text
            partition = 'train'
            # Use work_id as label for book-level topic analysis
            label = str(row['work_id'])
            tsv_data.append([sentence, partition, label])
        
        # Write the data to a TSV file
        with open(octis_corpus_path, mode='w', newline='', encoding='utf-8') as tsv_file:
            tsv_writer = csv.writer(tsv_file, delimiter='\t')
            for row in tsv_data:
                tsv_writer.writerow(row)
        
        logger.info(f"  ✓ Created OCTIS corpus: {octis_corpus_path}")
        logger.info(f"  Total sentences: {len(tsv_data):,}")
        logger.info(f"  Label format: work_id (book-level)")

        octis_dataset = Dataset()
        octis_dataset.load_custom_dataset_from_folder(str(octis_dataset_path))
        logger.info(f"  ✓ OCTIS dataset loaded")

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

                logger.info(f"Training #{trainings_count}")
                results_dir = DATA_INTERIM / 'octis_reviews' / 'optimization_results' / self.embedding_model_name
                results_dir.mkdir(parents=True, exist_ok=True)
                file_path = results_dir / 'result.json'
                results_csv_path = results_dir / 'results.csv'
                
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


                self.set_hyperparameters(hyperparameters)
                print("Training with parameters:")
                pprint.pprint(self.hyperparameters)

                # Preventing Stochastic Behavior
                # https://colab.research.google.com/drive/1BoQ_vakEVtojsd2x_U6-_x52OOuqruj2#scrollTo=28_EVoOfyZLb
                # Use GPU or CPU version based on availability
                if USE_GPU:
                    umap_model = GPU_UMAP(**self.hyperparameters['umap'])
                else:
                    umap_model = CPU_UMAP(**self.hyperparameters['umap'])

                # Controlling Number of Topics
                # https://colab.research.google.com/drive/1BoQ_vakEVtojsd2x_U6-_x52OOuqruj2#scrollTo=TH6vZPGU2zpg
                # Use GPU or CPU version based on availability
                if USE_GPU:
                    hdbscan_model = GPU_HDBSCAN(**self.hyperparameters['hdbscan'])
                else:
                    hdbscan_model = CPU_HDBSCAN(**self.hyperparameters['hdbscan'])

                # Improving Default Representation
                # https://colab.research.google.com/drive/1BoQ_vakEVtojsd2x_U6-_x52OOuqruj2#scrollTo=66zgeCyf0jy3&line=1&uniqifier=1
                vectorizer_model = CountVectorizer(**self.hyperparameters['vectorizer'])

                # Using ClassTfidfTransformer instead of CountVectorizer
                tfdf_model = ClassTfidfTransformer(**self.hyperparameters['tfdf_vectorizer'])

                topic_model = BERTopic(

                    # Pipeline models
                    embedding_model=self.embedding_model,
                    umap_model=umap_model,
                    hdbscan_model=hdbscan_model,
                    vectorizer_model=vectorizer_model,
                    representation_model=representation_model,

                    # Hyperparameters
                    **self.hyperparameters['bertopic']
                )

                try:
                    topics, probabilities = topic_model.fit_transform(dataset_as_list_of_strings, embeddings=self.embeddings)

                    # Calculate number of topics (excluding outliers with topic -1)
                    # According to BERTopic API: topics_ is a list of topic assignments
                    # https://maartengr.github.io/BERTopic/api/bertopic.html
                    unique_topics = set(topics)
                    n_topics = len([t for t in unique_topics if t != -1])  # Exclude outlier topic -1
                    
                    logger.info(f"  Number of topics found: {n_topics} (excluding outliers)")
                    
                    # Filter out models with less than 100 topics
                    MIN_TOPICS = 100
                    if n_topics < MIN_TOPICS:
                        logger.warning(f"  ⚠ Model rejected: Only {n_topics} topics found (minimum required: {MIN_TOPICS})")
                        print("Cleaning CUDA cache...")
                        torch.cuda.empty_cache()
                        gc.collect()
                        del topic_model
                        # Log to CSV before returning
                        try:
                            # Get current iteration from optimizer if available
                            iteration = getattr(self, '_current_iteration', trainings_count)
                            append_to_results_csv(iteration, self.hyperparameters, n_topics, accepted=False)
                        except Exception as e:
                            logger.warning(f"Failed to write to CSV: {e}")
                        # Return None so OCTIS doesn't process this result
                        output_dict = {'topics': None, 'n_topics': n_topics}
                        return output_dict

                    print("Cleaning CUDA cache...")
                    torch.cuda.empty_cache()
                    gc.collect()  # Force garbage collection

                    # cuda.select_device(0)
                    # cuda.close()
                    # cuda.select_device(0)

                    # Creating the required output dictionary
                    output_dict = {}
                    output_dict['topics'] = []
                    output_dict['n_topics'] = n_topics  # Store number of topics

                    # cleaned_docs = topic_model._preprocess_text(dataset_as_list_of_strings)
                    #
                    # # Extract vectorizer and analyzer from BERTopic
                    # vectorizer = topic_model.vectorizer_model
                    # analyzer = vectorizer.build_analyzer()
                    # tokens = [analyzer(doc) for doc in cleaned_docs]
                    dictionary = corpora.Dictionary(dataset_as_list_of_lists)

                    # https://github.com/MaartenGr/BERTopic/issues/90#issuecomment-820915389
                    for topic in range(len(set(topics)) - topic_model._outliers):
                        words = list(zip(*topic_model.get_topic(topic)))[0]
                        words = [word for word in words if word in dictionary.token2id]
                        words = [word for word in words if word.lower() != "mr"]
                        words = [word for word in words if word.lower() != "ms"]
                        output_dict['topics'].append(words)

                    output_dict['topics'] = [words for words in output_dict['topics'] if len(words) > 0]

                    # # 1. topics: list of most significant words for each topic
                    # for topic_number in set(topics):
                    #     if topic_number != -1:
                    #         topic_words = topic_model.get_topic(topic_number)
                    #         if topic_words:
                    #             output_dict['topics'].append(topic_words)
                    #
                    # output_dict['topics'] = [[word for word, _ in topic if word] for topic in output_dict['topics'] if
                    #                         all(word.strip() for word, _ in topic)]
                    #
                    # output_dict['topics'] = [words for words in output_dict['topics'] if len(words) > 0]


                    # If the array is empty, return None, so that OCTIS doesn't process this result
                    if not output_dict['topics']:
                        output_dict['topics'] = None

                    # 2. topic-word-matrix: c-TF-IDF matrix
                    output_dict['topic-word-matrix'] = np.array(topic_model.c_tf_idf_)

                    # 3. topic-document-matrix: probabilities matrix
                    output_dict['topic-document-matrix'] = np.array(probabilities)

                    logger.info(f"  ✓ Model accepted: {n_topics} topics found")
                    pprint.pprint(output_dict['topics'])
                    
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
                            self.hyperparameters[section][hyperparameter] = value
                        else:
                            print(f"Warning: Parameter '{key}' is not recognized.")
                    else:
                        print(f"Warning: Parameter '{key}' does not match the expected format 'section__hyperparameter'.")


        # Use only all-mpnet-base-v2 for initial testing
        embedding_model_names = [
            "all-mpnet-base-v2"
        ]
        
        logger.info(f"\nUsing embedding model: {embedding_model_names[0]}")

        embedding_models = []
        
        # Determine device for embeddings (GPU/CPU fallback)
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"  Using GPU device: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.warning("  GPU not available, using CPU for embeddings (will be slower)")

        for embedding_model_name in embedding_model_names:
            logger.info(f"Loading embedding model: {embedding_model_name} on {device}...")
            embedding_model = SentenceTransformer(embedding_model_name, device=device)
            embedding_models.append(embedding_model)
            logger.info(f"  ✓ Model loaded")

        # Store the pre-calculated embeddings
        # https://colab.research.google.com/drive/1BoQ_vakEVtojsd2x_U6-_x52OOuqruj2#scrollTo=sVfnYtUaxyLT
        precalculated_embeddings = []


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
        logger.info("\nCalculating embeddings (this may take a while)...")
        logger.info(f"  Using raw, unpreprocessed text for embeddings")
        logger.info(f"  Number of sentences: {len(dataset_as_list_of_strings):,}")
        
        embedding_file.parent.mkdir(parents=True, exist_ok=True)
        
        for idx, embedding_model in enumerate(embedding_models):
            embedding_model_name = embedding_model_names[idx]
            model_embedding_file = embedding_file.parent / f"embeddings_{embedding_model_name.replace('/', '_')}.pkl"
            
            # Check if embeddings already exist
            if model_embedding_file.exists():
                logger.info(f"  Loading existing embeddings from: {model_embedding_file}")
                embeddings = load_embeddings(str(model_embedding_file))
                precalculated_embeddings.append(embeddings)
            else:
                logger.info(f"  Calculating embeddings with {embedding_model_name}...")
                logger.info(f"    This may take several minutes for {len(dataset_as_list_of_strings):,} sentences...")
                start_time = time.time()
                embeddings = embedding_model.encode(
                    dataset_as_list_of_strings,
                    show_progress_bar=True,
                    batch_size=32,
                    convert_to_numpy=True
                )
                elapsed = time.time() - start_time
                logger.info(f"    ✓ Embeddings calculated in {elapsed/60:.1f} minutes")
                logger.info(f"    Embedding shape: {embeddings.shape}")
                
                # Save embeddings
                save_embeddings(embeddings, str(model_embedding_file))
                precalculated_embeddings.append(embeddings)

        def optimize_hyperparameters(model, label, dataset_as_list_of_lists):
            search_space = {
                'umap__n_neighbors': Integer(2, 50),
                'umap__n_components': Integer(2, 10),
                'umap__min_dist': Real(0.0, 0.1),
                'hdbscan__min_cluster_size': Integer(50, 500),
                'hdbscan__min_samples': Integer(10, 100),
                'vectorizer__min_df': Real(0.001, 0.01),
                'bertopic__top_n_words': Integer(10, 40),
                'bertopic__min_topic_size': Integer(10, 250)
            }

            # As a rule of thumb, if you have N hyperparameters to optimize, then you should make at least 15 times N iterations.
            # https://colab.research.google.com/github/MIND-Lab/OCTIS/blob/master/examples/OCTIS_Optimizing_CTM.ipynb#scrollTo=njjkNjl9CJW8&line=1&uniqifier=1
            optimization_runs = len(search_space) * 15 # should be 15

            # Topic models are usually probabilistic and thus produce different results even with the same hyperparameter configuration.
            # So we run the model multiple times and then take the median of the evaluated metric to get a more reliable result.
            model_runs = 1 # 1 is enough

            topk = 10

            dataset_as_list_of_lists = [doc for doc in dataset_as_list_of_lists if len(doc) > 0]

            # Use a different coherence measure
            npmi = Coherence(texts=dataset_as_list_of_lists, topk=topk, measure='c_v')
            diversity = TopicDiversity(topk=10)  # Initialize metric

            optimizer = Optimizer()

            results_path = str(DATA_INTERIM / 'octis_reviews' / 'optimization_results' / label) + '/'
            result_file = results_path + 'result.json'

            if os.path.isfile(result_file):
                logger.info(f"Resuming optimization from existing results: {result_file}")
                optimization_result = optimizer.resume_optimization(result_file, model=model)
                return optimization_result
            else:
                logger.info(f"Starting new optimization. Results will be saved to: {results_path}")
                optimization_result = optimizer.optimize(
                    model, octis_dataset, npmi, search_space, number_of_call=optimization_runs,
                    model_runs=model_runs, save_models=True,
                    extra_metrics=[diversity],  # to keep track of other metrics
                    save_path=results_path
                )
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


        def optimize_with_restart(model, embedding_model_name, dataset_as_list_of_lists, max_retries=3, delay=5):
            """
            Attempts to run optimize_hyperparameters and restarts if an exception occurs.

            :param model: The model to optimize
            :param embedding_model_name: Name of the embedding model
            :param dataset_as_list_of_lists: The dataset
            :param max_retries: Maximum number of retry attempts
            :param delay: Delay in seconds between retries
            :return: The optimization result or None if all attempts fail
            """
            for attempt in range(max_retries):
                try:
                    print(f"Attempt {attempt + 1} of {max_retries}")
                    optimization_result = optimize_hyperparameters(model, embedding_model_name,
                                                                   dataset_as_list_of_lists)
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

        for index, embedding_model in enumerate(embedding_models):
            try:
                embedding_model_name = embedding_model_names[index]
                print("Instantiating BERTopicOctisModelWithEmbeddings for ", embedding_model_name)
                model = BERTopicOctisModelWithEmbeddings(embedding_model=embedding_model,
                                                         embedding_model_name=embedding_model_names[index],
                                                        embeddings=precalculated_embeddings[index],
                                                        dataset_as_list_of_strings=dataset_as_list_of_strings)

                print("Optimizing hyperparameters for ", embedding_model_name)
                optimization_result = optimize_with_restart(model, embedding_model_name, dataset_as_list_of_lists)
                # optimization_results.append(optimization_result)
                print("=== Optimized hyperparameters for embedding_model_name:", embedding_model_name)

                # Clean up
                del model
                del optimization_result

            except Exception as ex:
                embedding_model_name = embedding_model_names[index]
                print(">>>>>>>>>>>>>>>>>>>>>****OPTIMIZATION ERROR****<<<<<<<<<<<<<<<<<<<")
                print("Model name:", embedding_model_name)
                print("Exception:")
                print(ex)
                print("==================================================================")
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
