#!/usr/bin/env python3
"""
Deep Topic Exploration Script

Analyzes BERTopic results to understand:
- Topic characteristics and representative words
- Book assignments and topic distributions
- Topic quality metrics and coherence
- Sample books for each topic
- Topic relationships and overlaps
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from bertopic import BERTopic


def load_topic_data(results_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, BERTopic]:
    """Load all topic modeling results and model."""
    
    # Load CSV results
    topic_info = pd.read_csv(os.path.join(results_dir, 'step2_topics', 'topic_info.csv'))
    book_topics = pd.read_csv(os.path.join(results_dir, 'step2_topics', 'book_topics.csv'))
    books_norm = pd.read_csv(os.path.join(results_dir, 'step1_normalized', 'books_with_normalized_shelves.csv'))
    
    # Load BERTopic model
    model_path = os.path.join(results_dir, 'step2_topics', 'bertopic_model')
    model = BERTopic.load(model_path)
    
    return topic_info, book_topics, books_norm, model


def analyze_topic_characteristics(topic_info: pd.DataFrame, book_topics: pd.DataFrame) -> None:
    """Analyze basic topic characteristics."""
    
    print("=" * 60)
    print("TOPIC CHARACTERISTICS ANALYSIS")
    print("=" * 60)
    
    # Basic stats
    total_topics = len(topic_info[topic_info['Topic'] != -1])
    outlier_count = (book_topics['topic'] == -1).sum()
    total_books = len(book_topics)
    
    print(f"Total Topics: {total_topics}")
    print(f"Outlier Books: {outlier_count} ({outlier_count/total_books*100:.1f}%)")
    print(f"Assigned Books: {total_books - outlier_count} ({(total_books - outlier_count)/total_books*100:.1f}%)")
    
    # Topic size distribution
    topic_sizes = topic_info[topic_info['Topic'] != -1]['Count'].values
    print(f"\nTopic Size Statistics:")
    print(f"  Mean: {topic_sizes.mean():.1f} books")
    print(f"  Median: {np.median(topic_sizes):.1f} books")
    print(f"  Min: {topic_sizes.min()} books")
    print(f"  Max: {topic_sizes.max()} books")
    print(f"  Std: {topic_sizes.std():.1f} books")


def explore_topic_words(model: BERTopic, topic_info: pd.DataFrame, top_n: int = 10) -> None:
    """Explore representative words for each topic."""
    
    print("\n" + "=" * 60)
    print("TOPIC REPRESENTATIVE WORDS")
    print("=" * 60)
    
    # Get topic representations
    topics = model.get_topics()
    
    for topic_id in sorted(topics.keys()):
        if topic_id == -1:  # Skip outliers
            continue
            
        # Get topic info
        topic_row = topic_info[topic_info['Topic'] == topic_id]
        if len(topic_row) == 0:
            continue
            
        count = topic_row['Count'].iloc[0]
        words = topics[topic_id][:top_n]  # Top N words
        
        print(f"\nTopic {topic_id} ({count} books):")
        word_str = ", ".join([f"{word} ({score:.3f})" for word, score in words])
        print(f"  Words: {word_str}")


def analyze_topic_books(book_topics: pd.DataFrame, books_norm: pd.DataFrame, 
                       topic_info: pd.DataFrame, model: BERTopic, 
                       books_per_topic: int = 5) -> None:
    """Analyze sample books for each topic."""
    
    print("\n" + "=" * 60)
    print("SAMPLE BOOKS BY TOPIC")
    print("=" * 60)
    
    # Merge with book data
    merged = book_topics.merge(books_norm[['work_id', 'title', 'author_name', 'normalized_shelves']], 
                              on='work_id', how='left')
    
    # Get topic representations for context
    topics = model.get_topics()
    
    for topic_id in sorted(merged['topic'].unique()):
        if topic_id == -1:  # Skip outliers
            continue
            
        topic_books = merged[merged['topic'] == topic_id].sort_values('prob', ascending=False)
        topic_row = topic_info[topic_info['Topic'] == topic_id]
        
        if len(topic_row) == 0:
            continue
            
        count = topic_row['Count'].iloc[0]
        top_words = [word for word, _ in topics[topic_id][:5]]
        
        print(f"\nTopic {topic_id} ({count} books) - Key words: {', '.join(top_words)}")
        print("-" * 50)
        
        for i, (_, book) in enumerate(topic_books.head(books_per_topic).iterrows()):
            shelves = json.loads(book['normalized_shelves']) if book['normalized_shelves'] else []
            print(f"  {i+1}. \"{book['title']}\" by {book['author_name']}")
            print(f"     Prob: {book['prob']:.3f} | Shelves: {', '.join(shelves[:8])}{'...' if len(shelves) > 8 else ''}")


def analyze_topic_coherence(book_topics: pd.DataFrame, books_norm: pd.DataFrame) -> None:
    """Analyze topic coherence and quality metrics."""
    
    print("\n" + "=" * 60)
    print("TOPIC COHERENCE ANALYSIS")
    print("=" * 60)
    
    # Merge with book data
    merged = book_topics.merge(books_norm[['work_id', 'normalized_shelves']], 
                              on='work_id', how='left')
    
    # Calculate coherence metrics per topic
    topic_metrics = []
    
    for topic_id in sorted(merged['topic'].unique()):
        if topic_id == -1:  # Skip outliers
            continue
            
        topic_books = merged[merged['topic'] == topic_id]
        
        # Collect all shelves for this topic
        all_shelves = []
        for shelves_str in topic_books['normalized_shelves']:
            if pd.notna(shelves_str):
                shelves = json.loads(shelves_str)
                all_shelves.extend(shelves)
        
        if not all_shelves:
            continue
            
        # Calculate metrics
        shelf_counts = Counter(all_shelves)
        unique_shelves = len(shelf_counts)
        total_shelves = len(all_shelves)
        avg_shelves_per_book = total_shelves / len(topic_books)
        
        # Top shelf concentration
        top_5_shelves = sum(count for _, count in shelf_counts.most_common(5))
        concentration = top_5_shelves / total_shelves if total_shelves > 0 else 0
        
        # Average probability
        avg_prob = topic_books['prob'].mean()
        
        topic_metrics.append({
            'topic_id': topic_id,
            'book_count': len(topic_books),
            'unique_shelves': unique_shelves,
            'avg_shelves_per_book': avg_shelves_per_book,
            'top_shelf_concentration': concentration,
            'avg_probability': avg_prob,
            'top_shelves': [shelf for shelf, _ in shelf_counts.most_common(5)]
        })
    
    # Sort by book count
    topic_metrics.sort(key=lambda x: x['book_count'], reverse=True)
    
    print(f"{'Topic':<6} {'Books':<6} {'AvgProb':<8} {'Concent':<8} {'Top Shelves'}")
    print("-" * 80)
    
    for metrics in topic_metrics:
        top_shelves_str = ", ".join(metrics['top_shelves'][:3])
        print(f"{metrics['topic_id']:<6} {metrics['book_count']:<6} "
              f"{metrics['avg_probability']:<8.3f} {metrics['top_shelf_concentration']:<8.3f} "
              f"{top_shelves_str}")


def analyze_topic_overlaps(book_topics: pd.DataFrame, books_norm: pd.DataFrame, 
                          model: BERTopic) -> None:
    """Analyze topic overlaps and relationships."""
    
    print("\n" + "=" * 60)
    print("TOPIC OVERLAP ANALYSIS")
    print("=" * 60)
    
    # Get topic representations
    topics = model.get_topics()
    
    # Calculate topic similarity matrix
    topic_ids = [tid for tid in topics.keys() if tid != -1]
    n_topics = len(topic_ids)
    
    if n_topics < 2:
        print("Not enough topics for overlap analysis")
        return
    
    # Get topic embeddings for similarity
    try:
        topic_embeddings = model.topic_embeddings_
        if topic_embeddings is not None:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Calculate cosine similarity between topics
            similarities = cosine_similarity(topic_embeddings)
            
            print("Most Similar Topic Pairs:")
            print("-" * 40)
            
            # Find top similar pairs
            pairs = []
            for i in range(n_topics):
                for j in range(i+1, n_topics):
                    sim = similarities[i, j]
                    pairs.append((topic_ids[i], topic_ids[j], sim))
            
            pairs.sort(key=lambda x: x[2], reverse=True)
            
            for topic1, topic2, sim in pairs[:5]:
                words1 = [word for word, _ in topics[topic1][:3]]
                words2 = [word for word, _ in topics[topic2][:3]]
                print(f"  Topics {topic1} & {topic2}: {sim:.3f}")
                print(f"    {topic1}: {', '.join(words1)}")
                print(f"    {topic2}: {', '.join(words2)}")
                print()
        else:
            print("Topic embeddings not available for similarity analysis")
            
    except Exception as e:
        print(f"Could not calculate topic similarities: {e}")


def main():
    parser = argparse.ArgumentParser(description="Explore BERTopic results in detail")
    parser.add_argument('--results-dir', required=True, 
                       help='Directory containing pipeline results (step1_normalized and step2_topics)')
    parser.add_argument('--books-per-topic', type=int, default=5,
                       help='Number of sample books to show per topic')
    parser.add_argument('--top-words', type=int, default=10,
                       help='Number of top words to show per topic')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading topic modeling results...")
    topic_info, book_topics, books_norm, model = load_topic_data(args.results_dir)
    
    # Run analyses
    analyze_topic_characteristics(topic_info, book_topics)
    explore_topic_words(model, topic_info, args.top_words)
    analyze_topic_books(book_topics, books_norm, topic_info, model, args.books_per_topic)
    analyze_topic_coherence(book_topics, books_norm)
    analyze_topic_overlaps(book_topics, books_norm, model)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
