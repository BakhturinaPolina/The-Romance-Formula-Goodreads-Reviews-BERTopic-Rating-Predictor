#!/usr/bin/env python3
"""
Topic Visualization Script

Creates visualizations for BERTopic results:
- Topic size distribution
- Topic coherence metrics
- Topic similarity heatmap
- Word clouds for top topics
"""

import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from bertopic import BERTopic
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity


def load_data(results_dir: str):
    """Load topic modeling results."""
    topic_info = pd.read_csv(os.path.join(results_dir, 'step2_topics', 'topic_info.csv'))
    book_topics = pd.read_csv(os.path.join(results_dir, 'step2_topics', 'book_topics.csv'))
    model_path = os.path.join(results_dir, 'step2_topics', 'bertopic_model')
    model = BERTopic.load(model_path)
    return topic_info, book_topics, model


def plot_topic_sizes(topic_info: pd.DataFrame, save_path: str = None):
    """Plot topic size distribution."""
    # Filter out outliers
    topics = topic_info[topic_info['Topic'] != -1].copy()
    
    plt.figure(figsize=(12, 6))
    
    # Bar plot of topic sizes
    plt.subplot(1, 2, 1)
    topics_sorted = topics.sort_values('Count', ascending=True)
    bars = plt.barh(range(len(topics_sorted)), topics_sorted['Count'])
    plt.yticks(range(len(topics_sorted)), [f"Topic {t}" for t in topics_sorted['Topic']])
    plt.xlabel('Number of Books')
    plt.title('Topic Size Distribution')
    plt.grid(axis='x', alpha=0.3)
    
    # Color bars by size
    colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Histogram of topic sizes
    plt.subplot(1, 2, 2)
    plt.hist(topics['Count'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Number of Books per Topic')
    plt.ylabel('Frequency')
    plt.title('Distribution of Topic Sizes')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_topic_coherence(book_topics: pd.DataFrame, save_path: str = None):
    """Plot topic coherence metrics."""
    # Calculate average probability per topic
    topic_probs = book_topics[book_topics['topic'] != -1].groupby('topic')['prob'].agg(['mean', 'std', 'count']).reset_index()
    topic_probs = topic_probs.sort_values('mean', ascending=True)
    
    plt.figure(figsize=(10, 6))
    
    # Bar plot with error bars
    bars = plt.barh(range(len(topic_probs)), topic_probs['mean'], 
                    xerr=topic_probs['std'], capsize=3, alpha=0.7)
    plt.yticks(range(len(topic_probs)), [f"Topic {t}" for t in topic_probs['topic']])
    plt.xlabel('Average Topic Probability')
    plt.title('Topic Coherence (Average Assignment Probability)')
    plt.grid(axis='x', alpha=0.3)
    
    # Color by book count
    colors = plt.cm.plasma(topic_probs['count'] / topic_probs['count'].max())
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, 
                               norm=plt.Normalize(vmin=topic_probs['count'].min(), 
                                                vmax=topic_probs['count'].max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Number of Books')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_topic_similarity(model: BERTopic, save_path: str = None):
    """Plot topic similarity heatmap."""
    try:
        # Get topic embeddings
        topic_embeddings = model.topic_embeddings_
        if topic_embeddings is None:
            print("Topic embeddings not available for similarity visualization")
            return
        
        # Calculate similarity matrix
        similarities = cosine_similarity(topic_embeddings)
        
        # Get topic IDs (excluding outliers)
        topics = model.get_topics()
        topic_ids = [tid for tid in topics.keys() if tid != -1]
        
        # Create similarity matrix for non-outlier topics
        topic_indices = [list(topics.keys()).index(tid) for tid in topic_ids]
        sim_matrix = similarities[np.ix_(topic_indices, topic_indices)]
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(sim_matrix, 
                   xticklabels=[f"T{t}" for t in topic_ids],
                   yticklabels=[f"T{t}" for t in topic_ids],
                   annot=True, fmt='.2f', cmap='viridis',
                   cbar_kws={'label': 'Cosine Similarity'})
        plt.title('Topic Similarity Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Could not create similarity heatmap: {e}")


def create_word_clouds(model: BERTopic, topic_info: pd.DataFrame, 
                      top_n_topics: int = 6, save_dir: str = None):
    """Create word clouds for top topics."""
    topics = model.get_topics()
    
    # Get top N topics by size
    top_topics = topic_info[topic_info['Topic'] != -1].nlargest(top_n_topics, 'Count')
    
    # Create subplots
    n_cols = 3
    n_rows = (top_n_topics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, (_, topic_row) in enumerate(top_topics.iterrows()):
        topic_id = topic_row['Topic']
        count = topic_row['Count']
        
        if topic_id in topics:
            # Create word cloud data
            word_freq = dict(topics[topic_id][:20])  # Top 20 words
            
            # Generate word cloud
            wordcloud = WordCloud(width=400, height=300, 
                                background_color='white',
                                colormap='viridis').generate_from_frequencies(word_freq)
            
            # Plot
            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].set_title(f'Topic {topic_id} ({count} books)', fontsize=12)
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(top_n_topics, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'topic_wordclouds.png'), 
                   dpi=300, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize BERTopic results")
    parser.add_argument('--results-dir', required=True,
                       help='Directory containing pipeline results')
    parser.add_argument('--output-dir', default='topic_visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--top-topics', type=int, default=6,
                       help='Number of top topics for word clouds')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading topic modeling results...")
    topic_info, book_topics, model = load_data(args.results_dir)
    
    # Create visualizations
    print("Creating topic size distribution plot...")
    plot_topic_sizes(topic_info, os.path.join(args.output_dir, 'topic_sizes.png'))
    
    print("Creating topic coherence plot...")
    plot_topic_coherence(book_topics, os.path.join(args.output_dir, 'topic_coherence.png'))
    
    print("Creating topic similarity heatmap...")
    plot_topic_similarity(model, os.path.join(args.output_dir, 'topic_similarity.png'))
    
    print("Creating word clouds for top topics...")
    create_word_clouds(model, topic_info, args.top_topics, args.output_dir)
    
    print(f"Visualizations saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
