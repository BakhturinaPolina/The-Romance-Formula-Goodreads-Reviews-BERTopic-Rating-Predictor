"""
Core preparation logic for BERTopic review corpus.
"""

from .prepare_bertopic_input import (
    load_spacy_model,
    extract_sentences_from_doc,
    split_reviews_to_sentences,
    clean_sentence_text,
    create_sentence_dataset,
    save_sentence_dataset,
    main
)

__all__ = [
    'load_spacy_model',
    'extract_sentences_from_doc',
    'split_reviews_to_sentences',
    'clean_sentence_text',
    'create_sentence_dataset',
    'save_sentence_dataset',
    'main'
]

