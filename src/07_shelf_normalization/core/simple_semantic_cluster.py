#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Semantic Clustering for Shelves - EASIEST Approach

Groups shelves by semantic meaning using keyword matching and simple rules:
- Emotions: funny, sad, hilarious, sweet, tear-jerker
- Tropes: forced-marriage, enemies-to-lovers, virgin-heroine, alpha-male
- Genre/Heat: erotica, hot, steamy, explicit
- Non-content: year, format, reading-status (already filtered)

This is the SIMPLEST approach - no ML, no embeddings, just keyword matching.

Usage:
    python simple_semantic_cluster.py --input shelf_canonical_cleaned.csv --output shelf_clusters.csv
"""

from __future__ import annotations
import argparse
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# SEMANTIC CATEGORIES - Keyword-based matching (EASIEST APPROACH)
# ============================================================================

# Emotion-related keywords
EMOTION_KEYWORDS: Dict[str, List[str]] = {
    'funny_humorous': ['funny', 'humor', 'humour', 'humorous', 'hilarious', 'comedy', 'comedic', 'laugh', 'laughing', 'witty', 'witty'],
    'sad_emotional': ['sad', 'sadness', 'emotional', 'tear', 'tears', 'cry', 'crying', 'cried', 'heartbreak', 'heartbreaking', 'heart-wrenching', 'tragic'],
    'sweet_romantic': ['sweet', 'sweetheart', 'sweetie', 'cute', 'adorable', 'fluffy', 'feel-good', 'heartwarming'],
    'angry_intense': ['angry', 'rage', 'furious', 'intense', 'intensity', 'passionate'],
    'happy_joyful': ['happy', 'joy', 'joyful', 'cheerful', 'uplifting', 'inspiring']
}

# Trope-related keywords
TROPE_KEYWORDS: Dict[str, List[str]] = {
    'forced_arranged_marriage': ['forced-marriage', 'forced-marry', 'arranged-marriage', 'arranged-marry', 'marriage-of-convenience', 'moc', 'unwilling-wife', 'forced-wife'],
    'enemies_to_lovers': ['enemies-to-lovers', 'enemies-to-lover', 'enemy-to-lover', 'hate-to-love', 'hate-to-lovers'],
    'friends_to_lovers': ['friends-to-lovers', 'friends-to-lover', 'friend-to-lover', 'best-friends'],
    'second_chance': ['second-chance', 'second-chances', 'reunion', 'reunited'],
    'virgin_heroine': ['virgin', 'virgin-heroine', 'virgin-hero', 'first-time', 'innocent'],
    'alpha_male': ['alpha', 'alpha-male', 'alpha-hero', 'possessive', 'dominant', 'alpha-does-not-share'],
    'billionaire': ['billionaire', 'billionaires', 'rich', 'wealthy', 'millionaire'],
    'mafia_crime': ['mafia', 'mafioso', 'crime', 'criminal', 'gangster', 'mob'],
    'small_town': ['small-town', 'smalltown', 'small-town-romance'],
    'fake_relationship': ['fake-relationship', 'fake-dating', 'pretend', 'pretend-relationship'],
    'age_gap': ['age-gap', 'older-man', 'younger-woman', 'older-woman', 'younger-man'],
    'single_parent': ['single-parent', 'single-mom', 'single-dad', 'single-mother', 'single-father'],
    'pregnancy': ['pregnancy', 'pregnant', 'baby', 'babies', 'expecting'],
    'amnesia': ['amnesia', 'memory-loss', 'forgot'],
    'secret_baby': ['secret-baby', 'secret-child', 'hidden-baby'],
    'love_triangle': ['love-triangle', 'triangle', 'love-square'],
    'reverse_harem': ['reverse-harem', 'rh', 'why-choose', 'multiple-partners'],
    'fated_mates': ['fated-mates', 'fated', 'mate', 'soulmate', 'destined']
}

# Genre/Heat level keywords
GENRE_HEAT_KEYWORDS: Dict[str, List[str]] = {
    'erotica': ['erotica', 'erotic', 'erotic-romance'],
    'hot_steamy': ['hot', 'steamy', 'steam', 'sizzling', 'scorching', 'sultry'],
    'explicit': ['explicit', 'explicit-content', 'graphic', 'graphic-sex'],
    'spicy': ['spicy', 'spice', 'heat', 'heated'],
    'sweet_clean': ['clean', 'sweet-romance', 'clean-romance', 'innocent', 'chaste', 'kisses-only'],
    'closed_door': ['closed-door', 'fade-to-black', 'ftb']
}

# Additional genre keywords (not heat-related)
GENRE_KEYWORDS: Dict[str, List[str]] = {
    'contemporary': ['contemporary', 'contemporary-romance', 'modern'],
    'historical': ['historical', 'historical-romance', 'regency', 'victorian', 'medieval'],
    'paranormal': ['paranormal', 'paranormal-romance', 'supernatural', 'vampire', 'werewolf', 'shifter', 'magic'],
    'fantasy': ['fantasy', 'fantasy-romance', 'epic-fantasy'],
    'sci_fi': ['sci-fi', 'science-fiction', 'sf', 'futuristic', 'space'],
    'suspense_thriller': ['suspense', 'thriller', 'mystery', 'romantic-suspense'],
    'western': ['western', 'cowboy', 'ranch', 'rodeo'],
    'sports': ['sports', 'sport', 'athlete', 'football', 'hockey', 'baseball'],
    'military': ['military', 'navy', 'army', 'marine', 'soldier', 'veteran'],
    'mc_motorcycle': ['mc', 'motorcycle', 'biker', 'club']
}

# Non-content (should already be filtered, but double-check)
NONCONTENT_KEYWORDS: Dict[str, List[str]] = {
    'reading_status': ['to-read', 'currently-reading', 'read', 'finished', 'dnf', 'tbr', 'wtr'],
    'format': ['ebook', 'kindle', 'audiobook', 'hardcover', 'paperback', 'pdf', 'epub'],
    'year': ['read-2012', 'read-2013', 'read-2014', '2012-reads', '2013-reads'],
    'ownership': ['owned', 'library', 'borrowed', 'wishlist', 'i-own']
}


def match_keywords(shelf: str, keyword_dict: Dict[str, List[str]]) -> Optional[str]:
    """
    Match shelf to category based on keywords.
    Returns category name if match found, None otherwise.
    """
    shelf_lower = shelf.lower()
    
    # Check each category
    for category, keywords in keyword_dict.items():
        for keyword in keywords:
            # Match whole word or as part of hyphenated word
            pattern = r'\b' + re.escape(keyword) + r'\b|' + re.escape(keyword) + r'[-_]'
            if re.search(pattern, shelf_lower):
                return category
    
    return None


def cluster_shelves_semantic(
    canon_df: pd.DataFrame,
    min_frequency: int = 3,
    verbose: bool = False
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Cluster shelves by semantic meaning using keyword matching.
    
    Args:
        canon_df: DataFrame with 'shelf_canon' and 'count' columns
        min_frequency: Minimum frequency to include shelf
        verbose: Whether to print detailed progress
        
    Returns:
        Tuple of (clustered DataFrame, statistics dict)
    """
    stats = {
        'total_shelves': len(canon_df),
        'filtered_by_frequency': 0,
        'emotion_clusters': defaultdict(int),
        'trope_clusters': defaultdict(int),
        'genre_heat_clusters': defaultdict(int),
        'genre_clusters': defaultdict(int),
        'noncontent_clusters': defaultdict(int),
        'unclustered': 0,
        'cluster_examples': defaultdict(list)  # Store examples for each cluster
    }
    
    # Filter by frequency
    if min_frequency > 1:
        before = len(canon_df)
        canon_df = canon_df[canon_df['count'] >= min_frequency].copy()
        stats['filtered_by_frequency'] = len(canon_df)
        if verbose:
            logger.info(f"  Filtered {before - len(canon_df):,} shelves below frequency {min_frequency}")
    
    # Initialize cluster columns
    canon_df['cluster_category'] = None
    canon_df['cluster_name'] = None
    canon_df['cluster_type'] = None
    
    if verbose:
        logger.info(f"  Processing {len(canon_df):,} shelves...")
        logger.info("  Checking categories in order: noncontent → emotion → trope → genre_heat → genre")
    
    # Cluster shelves
    processed = 0
    category_counts = defaultdict(int)
    
    if verbose:
        logger.info("")
        logger.info("  Starting clustering process...")
        logger.info("  Category checking order:")
        logger.info("    1. Non-content (reading status, format, year, ownership)")
        logger.info("    2. Emotions (funny, sad, sweet, etc.)")
        logger.info("    3. Tropes (forced-marriage, enemies-to-lovers, etc.)")
        logger.info("    4. Genre/Heat (erotica, hot, steamy, etc.)")
        logger.info("    5. Genre (contemporary, historical, paranormal, etc.)")
        logger.info("    6. Unclustered (if no match found)")
        logger.info("")
    
    for idx, row in canon_df.iterrows():
        shelf = row['shelf_canon']
        count = row['count']
        processed += 1
        
        if verbose and processed % 5000 == 0:
            logger.info(f"    Progress: {processed:,}/{len(canon_df):,} shelves ({processed/len(canon_df)*100:.1f}%)")
            logger.info(f"      Current stats: {sum(category_counts.values()):,} clustered, {stats['unclustered']:,} unclustered")
        
        matched = False
        
        # Check non-content first (should be filtered, but double-check)
        if verbose and processed <= 20:
            logger.info(f"    [{processed}] Checking shelf: '{shelf}' (count={count:,})")
        
        category = match_keywords(shelf, NONCONTENT_KEYWORDS)
        if category:
            canon_df.at[idx, 'cluster_category'] = category
            canon_df.at[idx, 'cluster_name'] = f'noncontent_{category}'
            canon_df.at[idx, 'cluster_type'] = 'noncontent'
            stats['noncontent_clusters'][category] += count
            category_counts['noncontent'] += count
            if verbose:
                if processed <= 20:
                    logger.info(f"      → Matched NONCONTENT: {category}")
                if len(stats['cluster_examples'][f'noncontent_{category}']) < 5:
                    stats['cluster_examples'][f'noncontent_{category}'].append((shelf, count))
            matched = True
        
        # Check emotions
        if not matched:
            category = match_keywords(shelf, EMOTION_KEYWORDS)
            if category:
                canon_df.at[idx, 'cluster_category'] = category
                canon_df.at[idx, 'cluster_name'] = f'emotion_{category}'
                canon_df.at[idx, 'cluster_type'] = 'emotion'
                stats['emotion_clusters'][category] += count
                category_counts['emotion'] += count
                if verbose:
                    if processed <= 20:
                        logger.info(f"      → Matched EMOTION: {category}")
                    if len(stats['cluster_examples'][f'emotion_{category}']) < 5:
                        stats['cluster_examples'][f'emotion_{category}'].append((shelf, count))
                matched = True
        
        # Check tropes
        if not matched:
            category = match_keywords(shelf, TROPE_KEYWORDS)
            if category:
                canon_df.at[idx, 'cluster_category'] = category
                canon_df.at[idx, 'cluster_name'] = f'trope_{category}'
                canon_df.at[idx, 'cluster_type'] = 'trope'
                stats['trope_clusters'][category] += count
                category_counts['trope'] += count
                if verbose:
                    if processed <= 20:
                        logger.info(f"      → Matched TROPE: {category}")
                    if len(stats['cluster_examples'][f'trope_{category}']) < 5:
                        stats['cluster_examples'][f'trope_{category}'].append((shelf, count))
                matched = True
        
        # Check genre/heat
        if not matched:
            category = match_keywords(shelf, GENRE_HEAT_KEYWORDS)
            if category:
                canon_df.at[idx, 'cluster_category'] = category
                canon_df.at[idx, 'cluster_name'] = f'genre_heat_{category}'
                canon_df.at[idx, 'cluster_type'] = 'genre_heat'
                stats['genre_heat_clusters'][category] += count
                category_counts['genre_heat'] += count
                if verbose:
                    if processed <= 20:
                        logger.info(f"      → Matched GENRE/HEAT: {category}")
                    if len(stats['cluster_examples'][f'genre_heat_{category}']) < 5:
                        stats['cluster_examples'][f'genre_heat_{category}'].append((shelf, count))
                matched = True
        
        # Check genre (non-heat)
        if not matched:
            category = match_keywords(shelf, GENRE_KEYWORDS)
            if category:
                canon_df.at[idx, 'cluster_category'] = category
                canon_df.at[idx, 'cluster_name'] = f'genre_{category}'
                canon_df.at[idx, 'cluster_type'] = 'genre'
                stats['genre_clusters'][category] += count
                category_counts['genre'] += count
                if verbose:
                    if processed <= 20:
                        logger.info(f"      → Matched GENRE: {category}")
                    if len(stats['cluster_examples'][f'genre_{category}']) < 5:
                        stats['cluster_examples'][f'genre_{category}'].append((shelf, count))
                matched = True
        
        # Unclustered
        if not matched:
            canon_df.at[idx, 'cluster_type'] = 'unclustered'
            stats['unclustered'] += count
            category_counts['unclustered'] += count
            if verbose and processed <= 20:
                logger.info(f"      → No match found (UNCLUSTERED)")
    
    if verbose:
        logger.info("")
        logger.info(f"  Completed processing {processed:,} shelves")
        logger.info("")
        logger.info("  CLUSTERING STATISTICS BY TYPE:")
        logger.info(f"    Non-content: {category_counts['noncontent']:,} occurrences")
        logger.info(f"    Emotions: {category_counts['emotion']:,} occurrences")
        logger.info(f"    Tropes: {category_counts['trope']:,} occurrences")
        logger.info(f"    Genre/Heat: {category_counts['genre_heat']:,} occurrences")
        logger.info(f"    Genre: {category_counts['genre']:,} occurrences")
        logger.info(f"    Unclustered: {category_counts['unclustered']:,} occurrences")
        logger.info("")
        logger.info("  DETAILED CLUSTER COUNTS:")
        logger.info(f"    Total emotion categories found: {len(stats['emotion_clusters'])}")
        logger.info(f"    Total trope categories found: {len(stats['trope_clusters'])}")
        logger.info(f"    Total genre/heat categories found: {len(stats['genre_heat_clusters'])}")
        logger.info(f"    Total genre categories found: {len(stats['genre_clusters'])}")
        logger.info(f"    Total non-content categories found: {len(stats['noncontent_clusters'])}")
    
    return canon_df, stats


def main():
    parser = argparse.ArgumentParser(
        description="Simple semantic clustering of shelves using keyword matching"
    )
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input shelf_canonical_cleaned.csv file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output shelf_clusters.csv file'
    )
    parser.add_argument(
        '--min-frequency',
        type=int,
        default=3,
        help='Minimum frequency to include shelf (default: 3)'
    )
    parser.add_argument(
        '--stats',
        type=Path,
        help='Output statistics JSON file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress'
    )
    
    args = parser.parse_args()
    
    # Load input
    logger.info(f"Loading {args.input}")
    canon_df = pd.read_csv(args.input, dtype=str)
    
    if 'shelf_canon' not in canon_df.columns:
        raise ValueError("Input must have 'shelf_canon' column")
    if 'count' not in canon_df.columns:
        canon_df['count'] = 1
    
    canon_df['count'] = pd.to_numeric(canon_df['count'], errors='coerce').fillna(1).astype(int)
    
    logger.info(f"Loaded {len(canon_df):,} shelves")
    
    # Print input statistics
    logger.info("=" * 60)
    logger.info("INPUT STATISTICS:")
    logger.info(f"  Total shelves: {len(canon_df):,}")
    logger.info(f"  Total occurrences: {canon_df['count'].sum():,}")
    logger.info(f"  Min frequency: {canon_df['count'].min()}")
    logger.info(f"  Max frequency: {canon_df['count'].max()}")
    logger.info(f"  Mean frequency: {canon_df['count'].mean():.1f}")
    logger.info("=" * 60)
    logger.info("")
    
    # Cluster
    logger.info("Clustering shelves by semantic meaning...")
    clustered_df, stats = cluster_shelves_semantic(canon_df, min_frequency=args.min_frequency, verbose=args.verbose)
    
    # Save output
    logger.info(f"Saving clustered shelves to {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    clustered_df.to_csv(args.output, index=False)
    
    # Save statistics
    if args.stats:
        import json
        # Convert defaultdict to dict for JSON
        # Convert cluster_examples to serializable format
        cluster_examples_serializable = {}
        for key, examples in stats['cluster_examples'].items():
            cluster_examples_serializable[key] = [(shelf, int(count)) for shelf, count in examples]
        
        stats_json = {
            'total_shelves': stats['total_shelves'],
            'filtered_by_frequency': stats['filtered_by_frequency'],
            'emotion_clusters': dict(stats['emotion_clusters']),
            'trope_clusters': dict(stats['trope_clusters']),
            'genre_heat_clusters': dict(stats['genre_heat_clusters']),
            'genre_clusters': dict(stats['genre_clusters']),
            'noncontent_clusters': dict(stats['noncontent_clusters']),
            'unclustered': stats['unclustered'],
            'cluster_examples': cluster_examples_serializable
        }
        with open(args.stats, 'w') as f:
            json.dump(stats_json, f, indent=2)
        logger.info(f"Statistics saved to {args.stats}")
    
    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("CLUSTERING SUMMARY:")
    logger.info("=" * 60)
    logger.info(f"  Total shelves: {stats['total_shelves']:,}")
    logger.info(f"  After frequency filter: {stats['filtered_by_frequency']:,}")
    
    # Count clustered vs unclustered
    clustered_count = sum([
        sum(stats['emotion_clusters'].values()),
        sum(stats['trope_clusters'].values()),
        sum(stats['genre_heat_clusters'].values()),
        sum(stats['genre_clusters'].values()),
        sum(stats['noncontent_clusters'].values())
    ])
    total_occurrences = clustered_count + stats['unclustered']
    cluster_rate = (clustered_count / total_occurrences * 100) if total_occurrences > 0 else 0
    
    logger.info(f"  Clustered occurrences: {clustered_count:,} ({cluster_rate:.1f}%)")
    logger.info(f"  Unclustered occurrences: {stats['unclustered']:,} ({100-cluster_rate:.1f}%)")
    logger.info("")
    
    # Emotion clusters
    if stats['emotion_clusters']:
        logger.info("  EMOTION CLUSTERS:")
        for cat, count in sorted(stats['emotion_clusters'].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"    - {cat}: {count:,} occurrences")
            if args.verbose and cat in [k.replace('emotion_', '') for k in stats['cluster_examples'].keys() if k.startswith('emotion_')]:
                examples = stats['cluster_examples'].get(f'emotion_{cat}', [])
                for shelf, shelf_count in examples[:3]:
                    logger.info(f"        Example: '{shelf}' (count={shelf_count:,})")
        logger.info("")
    
    # Trope clusters
    if stats['trope_clusters']:
        logger.info("  TROPE CLUSTERS:")
        for cat, count in sorted(stats['trope_clusters'].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"    - {cat}: {count:,} occurrences")
            if args.verbose and cat in [k.replace('trope_', '') for k in stats['cluster_examples'].keys() if k.startswith('trope_')]:
                examples = stats['cluster_examples'].get(f'trope_{cat}', [])
                for shelf, shelf_count in examples[:3]:
                    logger.info(f"        Example: '{shelf}' (count={shelf_count:,})")
        logger.info("")
    
    # Genre/Heat clusters
    if stats['genre_heat_clusters']:
        logger.info("  GENRE/HEAT CLUSTERS:")
        for cat, count in sorted(stats['genre_heat_clusters'].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"    - {cat}: {count:,} occurrences")
            if args.verbose and cat in [k.replace('genre_heat_', '') for k in stats['cluster_examples'].keys() if k.startswith('genre_heat_')]:
                examples = stats['cluster_examples'].get(f'genre_heat_{cat}', [])
                for shelf, shelf_count in examples[:3]:
                    logger.info(f"        Example: '{shelf}' (count={shelf_count:,})")
        logger.info("")
    
    # Genre clusters
    if stats['genre_clusters']:
        logger.info("  GENRE CLUSTERS:")
        for cat, count in sorted(stats['genre_clusters'].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"    - {cat}: {count:,} occurrences")
            if args.verbose and cat in [k.replace('genre_', '') for k in stats['cluster_examples'].keys() if k.startswith('genre_') and not k.startswith('genre_heat_')]:
                examples = stats['cluster_examples'].get(f'genre_{cat}', [])
                for shelf, shelf_count in examples[:3]:
                    logger.info(f"        Example: '{shelf}' (count={shelf_count:,})")
        logger.info("")
    
    # Non-content clusters
    if stats['noncontent_clusters']:
        logger.info("  NON-CONTENT CLUSTERS (should be minimal):")
        for cat, count in sorted(stats['noncontent_clusters'].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"    - {cat}: {count:,} occurrences")
        logger.info("")
    
    logger.info(f"  UNCLUSTERED: {stats['unclustered']:,} occurrences")
    logger.info("=" * 60)
    
    # Show detailed examples
    if args.verbose:
        logger.info("")
        logger.info("=" * 60)
        logger.info("DETAILED CLUSTER EXAMPLES:")
        logger.info("=" * 60)
        
        for cluster_type in ['emotion', 'trope', 'genre_heat', 'genre']:
            type_df = clustered_df[clustered_df['cluster_type'] == cluster_type]
            if len(type_df) > 0:
                logger.info(f"\n  {cluster_type.upper()} CLUSTERS (showing top 15 shelves):")
                # Group by cluster and show top shelves
                for cluster_name in type_df['cluster_name'].unique()[:10]:
                    cluster_shelves = type_df[type_df['cluster_name'] == cluster_name].nlargest(5, 'count')
                    if len(cluster_shelves) > 0:
                        logger.info(f"\n    Cluster: {cluster_name}")
                        for _, row in cluster_shelves.iterrows():
                            logger.info(f"      - '{row['shelf_canon']}' (count={row['count']:,})")
        
        # Show some unclustered examples
        unclustered = clustered_df[clustered_df['cluster_type'] == 'unclustered'].nlargest(20, 'count')
        if len(unclustered) > 0:
            logger.info(f"\n  UNCLUSTERED EXAMPLES (top 20 by frequency):")
            for _, row in unclustered.iterrows():
                logger.info(f"    - '{row['shelf_canon']}' (count={row['count']:,})")
        
        logger.info("")
        logger.info("=" * 60)


if __name__ == '__main__':
    main()

