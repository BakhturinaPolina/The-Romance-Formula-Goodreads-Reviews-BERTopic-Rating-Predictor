"""
Sub-dataset Sampling: Create 6,000 book corpus (2,000 per tier)
Based on the research methodology for balanced representation across popularity tiers.
"""

import pandas as pd
import numpy as np
import os
import sys
from scipy.stats import chisquare

# Add the project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def create_subdataset_6000(input_csv_path, output_csv_path):
    """
    Create a 6,000 book sub-dataset with equal representation across popularity tiers.
    
    Args:
        input_csv_path: Path to the main processed CSV file
        output_csv_path: Path where the sub-dataset should be saved
    """
    print("Loading main dataset...")
    meta = pd.read_csv(input_csv_path)
    print(f"Loaded {len(meta)} books")
    
    # Remove comics/graphic books from main dataset before processing
    print("Removing comics/graphic books from main dataset...")
    meta = meta[~meta['genres_str'].str.contains('comics|graphic', case=False, na=False)]
    print(f"After removing comics/graphic: {len(meta)} books")
    
    # Apply genre normalization to main dataset
    def canonicalize_genre(genre_str):
        """Enhanced canonicalization with historical romance merging"""
        if not isinstance(genre_str, str) or not genre_str.strip():
            return ""

        # Split by comma and process each genre
        genres = [g.strip().lower() for g in genre_str.split(',') if g.strip()]

        # Enhanced canonicalization with historical romance merging
        canonical_genres = []
        for genre in genres:
            # Normalize common variations
            if genre in ['sci fi', 'sci-fi', 'science fiction']:
                canonical_genres.append('science fiction')
            elif genre in ['ya', 'young adult']:
                canonical_genres.append('young adult')
            elif genre in ['historical fiction', 'historical', 'history']:
                canonical_genres.append('historical romance')  # Merge historical fiction and history
            elif genre in ['paranormal romance', 'paranormal']:
                canonical_genres.append('paranormal')
            elif genre in ['contemporary romance', 'contemporary']:
                canonical_genres.append('contemporary')
            elif genre in ['fantasy romance', 'fantasy']:
                canonical_genres.append('fantasy')
            elif genre in ['mystery romance', 'mystery']:
                canonical_genres.append('mystery')
            else:
                canonical_genres.append(genre)

        return ', '.join(sorted(set(canonical_genres)))
    
    # Apply genre canonicalization to main dataset
    meta['genres_str'] = meta['genres_str'].apply(canonicalize_genre)
    
    # Fix series flags
    meta['series_flag'] = (meta['series_id'].astype(str) != 'stand_alone').astype(int)
    
    # 2) Popularity tiers by quantiles computed on *your* file
    q1 = meta['average_rating_weighted_mean'].quantile(0.25)  # ~3.71 per your summary
    q3 = meta['average_rating_weighted_mean'].quantile(0.75)  # ~4.15 per your summary

    def tierize(x, lo=q1, hi=q3):
        if x < lo: return "thrash"
        if x > hi: return "top"
        return "mid"

    meta['pop_tier'] = meta['average_rating_weighted_mean'].apply(tierize)
    
    print(f"Tier boundaries: thrash < {q1:.3f}, mid ∈ [{q1:.3f}, {q3:.3f}], top > {q3:.3f}")
    print("Tier distribution in full dataset:")
    print(meta['pop_tier'].value_counts().sort_index())

    # 3) Derive representativeness features from allowed columns
    meta['pages_bin']   = pd.qcut(meta['num_pages_median'], q=4, labels=['Q1','Q2','Q3','Q4'])

    def genre_group(s: str):
        s = str(s).lower()
        if 'paranormal' in s:      return 'paranormal'
        if 'historical romance' in s: return 'historical'  # Updated for canonicalized genre
        if 'fantasy' in s:         return 'fantasy'
        if 'mystery' in s:         return 'mystery'
        if 'young adult' in s:     return 'young_adult'
        return 'other'
    meta['genre_group'] = meta['genres_str'].apply(genre_group)

    # 4) Target size
    TOTAL = 6000
    PER_TIER = TOTAL // 3  # 2000 each

    # 5) Within-tier quotas over (year x genre_group)
    def tier_quota_table(df_tier: pd.DataFrame, tier_target: int):
        base = (df_tier
                .groupby(['publication_year','genre_group'])
                .size()
                .rename('n')
                .reset_index())
        base['prop']  = base['n'] / base['n'].sum()
        base['quota'] = np.floor(base['prop'] * tier_target).astype(int)
        # ensure at least 1 where present
        base.loc[(base['n']>0) & (base['quota']==0), 'quota'] = 1
        # rounding fix to hit exact tier_target
        diff = tier_target - base['quota'].sum()
        if diff > 0:
            bump_idx = base.sort_values('prop', ascending=False).index[:diff]
            base.loc[bump_idx, 'quota'] += 1
        elif diff < 0:
            drop = base[base['quota']>1].sort_values('prop').index[:(-diff)]
            base.loc[drop, 'quota'] -= 1
        return base

    tier_tables = {t: tier_quota_table(meta[meta['pop_tier']==t], PER_TIER)
                   for t in ['thrash','mid','top']}

    # 6) Engagement ranking inside each tier
    meta = meta.sort_values(
        by=['pop_tier','ratings_count_sum','text_reviews_count_sum','author_ratings_count'],
        ascending=[True, False, False, False]
    )

    # 7) Selection per cell with soft preservation of series/pages composition
    def pick_from_cell(df_cell: pd.DataFrame, n: int) -> pd.DataFrame:
        if n <= 0 or df_cell.empty:
            return df_cell.iloc[0:0]
        inner = df_cell.copy()
        # desired mix by (series_flag, pages_bin)
        target = (inner.groupby(['series_flag','pages_bin']).size()
                  .rename('n').reset_index())
        target['prop']  = target['n'] / target['n'].sum()
        target['quota'] = np.floor(target['prop'] * n).astype(int)
        # rounding
        diff = n - target['quota'].sum()
        if diff > 0:
            bump = target.sort_values('prop', ascending=False).index[:diff]
            target.loc[bump, 'quota'] += 1

        parts = []
        for _, r in target.iterrows():
            sub = inner[(inner['series_flag']==r['series_flag']) & (inner['pages_bin']==r['pages_bin'])]
            parts.append(sub.head(int(r['quota'])))
        out = pd.concat(parts).drop_duplicates(subset=['work_id'])

        # top up if short
        if len(out) < n:
            remainder = inner[~inner['work_id'].isin(out['work_id'])]
            out = pd.concat([out, remainder.head(n - len(out))])
        return out.head(n)

    selected = []
    for t in ['thrash','mid','top']:
        df_t = meta[meta['pop_tier']==t]
        quotas = tier_tables[t]
        print(f"\nProcessing {t} tier...")
        for _, row in quotas.iterrows():
            yr, gg, q = int(row['publication_year']), row['genre_group'], int(row['quota'])
            cell = df_t[(df_t['publication_year']==yr) & (df_t['genre_group']==gg)]
            if not cell.empty:
                selected.append(pick_from_cell(cell, q))

    sample_df = (pd.concat(selected, ignore_index=True)
                 .drop_duplicates(subset=['work_id']))

    # 8) Backfill within tier to reach exact 2,000 each (in engagement order)
    def backfill_tier(sample_df, meta, tier, target_size):
        cur = sample_df[sample_df['pop_tier']==tier]
        need = target_size - len(cur)
        if need <= 0: return sample_df
        cand = meta[(meta['pop_tier']==tier) & (~meta['work_id'].isin(sample_df['work_id']))]
        cand = cand.sort_values(
            by=['ratings_count_sum','text_reviews_count_sum','author_ratings_count'],
            ascending=[False, False, False]
        )
        return pd.concat([sample_df, cand.head(need)], ignore_index=True)

    for t in ['thrash','mid','top']:
        sample_df = backfill_tier(sample_df, meta, t, PER_TIER)

    # Ensure exactly 6,000 by trimming lowest-engagement overflow if rounding exceeded
    if len(sample_df) > TOTAL:
        sample_df = (sample_df
                     .sort_values(['ratings_count_sum','text_reviews_count_sum','author_ratings_count'],
                                  ascending=[False, False, False])
                     .head(TOTAL))

    print(f"\nFinal shape: {sample_df.shape}")
    print("Final tier distribution:")
    print(sample_df['pop_tier'].value_counts().sort_index())

    # 9) Export the book list with key columns for downstream linking (texts/reviews)
    cols_out = ['work_id','title','author_id','author_name','publication_year',
                'num_pages_median','genres_str','series_id','series_title',
                'ratings_count_sum','text_reviews_count_sum',
                'average_rating_weighted_mean','pop_tier']
    sample_df[cols_out].to_csv(output_csv_path, index=False)
    
    # 9.5) Save updated main dataset with canonicalized genres and fixed series flags
    print(f"\nSaving updated main dataset with canonicalized genres...")
    main_output_path = input_csv_path.replace('.csv', '_canonicalized.csv')
    meta[cols_out[:-1]].to_csv(main_output_path, index=False)  # Exclude pop_tier from main dataset
    print(f"Updated main dataset saved: {main_output_path}")

    # 10) Enhanced representativeness validation with genre canonicalization
    def pct_table(df, col):
        vc = df[col].value_counts(normalize=True).sort_index()
        return (vc*100).round(2)

    print("\n" + "="*80)
    print("REPRESENTATIVENESS VALIDATION")
    print("="*80)

    for t in ['thrash','mid','top']:
        src = meta[meta['pop_tier']==t]
        dst = sample_df[sample_df['pop_tier']==t]
        print(f"\n=== Tier {t.upper()}: distribution checks ===")
        print("Year (%):\n", pd.concat({'Full': pct_table(src,'publication_year'),
                                         'Sample': pct_table(dst,'publication_year')}, axis=1).fillna(0))
        print("\nGenre_group (%):\n", pd.concat({'Full': pct_table(src,'genre_group'),
                                                'Sample': pct_table(dst,'genre_group')}, axis=1).fillna(0))
        print("\nSeries_flag (%):\n", pd.concat({'Full': pct_table(src,'series_flag'),
                                                 'Sample': pct_table(dst,'series_flag')}, axis=1).fillna(0))
        print("\nPages_bin (%):\n", pd.concat({'Full': pct_table(src,'pages_bin'),
                                              'Sample': pct_table(dst,'pages_bin')}, axis=1).fillna(0))

    # 11) Enhanced genre analysis (genres already canonicalized in main dataset)
    # Apply same canonicalization to sub dataset
    sub_canonical = sample_df.copy()
    sub_canonical['genres_str'] = sub_canonical['genres_str'].apply(canonicalize_genre)
    
    # Use canonicalized versions
    meta_canonical = meta.copy()
    meta_canonical['genres_canonical'] = meta_canonical['genres_str']
    sub_canonical['genres_canonical'] = sub_canonical['genres_str']

    # Count unique canonical genres
    main_genres = set()
    for genres in meta_canonical['genres_canonical']:
        if genres:
            main_genres.update([g.strip() for g in genres.split(',')])

    sub_genres = set()
    for genres in sub_canonical['genres_canonical']:
        if genres:
            sub_genres.update([g.strip() for g in genres.split(',')])

    print("\n=== GENRE CANONICALIZATION ANALYSIS ===")
    print(f"Main dataset unique canonical genres: {len(main_genres)}")
    print(f"Sub dataset unique canonical genres: {len(sub_genres)}")
    print(f"Genres in main but not in sub: {len(main_genres - sub_genres)}")
    print(f"Genres in sub but not in main: {len(sub_genres - main_genres)}")

    # Enhanced genre distribution comparison (excluding generic and weird genres)
    def get_genre_counts(df, exclude_generic=True, exclude_weird=True):
        genre_counts = {}
        for genres in df['genres_canonical']:
            if genres:
                for genre in genres.split(','):
                    genre = genre.strip()
                    # Exclude generic genres that are not informative
                    if exclude_generic and genre in ['romance', 'fiction']:
                        continue
                    # Exclude weird genres from distribution analysis
                    if exclude_weird and genre in ['biography', 'poetry', 'children']:
                        continue
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
        return genre_counts

    main_genre_counts = get_genre_counts(meta_canonical, exclude_generic=True, exclude_weird=True)
    sub_genre_counts = get_genre_counts(sub_canonical, exclude_generic=True, exclude_weird=True)

    # Sort by frequency in main dataset
    top_genres = sorted(main_genre_counts.items(), key=lambda x: x[1], reverse=True)[:20]

    print("\nCore romance genres comparison (excluding 'romance', 'fiction', 'biography', 'poetry', 'children'):")
    print(f"{'Genre'"<25"} {'Main_Count'"<12"} {'Main_%'"<10"} {'Sub_Count'"<12"} {'Sub_%'"<10"} {'Ratio'"<8"}")
    print("-" * 80)

    for genre, main_count in top_genres:
        sub_count = sub_genre_counts.get(genre, 0)
        main_pct = (main_count / len(meta_canonical)) * 100
        sub_pct = (sub_count / len(sub_canonical)) * 100
        ratio = sub_pct / main_pct if main_pct > 0 else 0

        print(f"{genre:25} {main_count:12} {main_pct:10.2f} {sub_count:12} {sub_pct:10.2f} {ratio:8.3f}")

    # Special analysis for excluded genres (kept in dataset but not in analysis)
    excluded_genres = ['biography', 'children', 'poetry']
    print(f"\n=== EXCLUDED GENRES (kept in dataset, excluded from analysis) ===")
    print(f"{'Genre'"<15"} {'Main_Count'"<12"} {'Main_%'"<10"} {'Sub_Count'"<12"} {'Sub_%'"<10"} {'Ratio'"<8"}")
    print("-" * 70)
    
    # Get counts including excluded genres for this analysis
    main_all_counts = get_genre_counts(meta_canonical, exclude_generic=True, exclude_weird=False)
    sub_all_counts = get_genre_counts(sub_canonical, exclude_generic=True, exclude_weird=False)
    
    for genre in excluded_genres:
        main_count = main_all_counts.get(genre, 0)
        sub_count = sub_all_counts.get(genre, 0)
        main_pct = (main_count / len(meta_canonical)) * 100
        sub_pct = (sub_count / len(sub_canonical)) * 100
        ratio = sub_pct / main_pct if main_pct > 0 else 0
        
        print(f"{genre:15} {main_count:12} {main_pct:10.2f} {sub_count:12} {sub_pct:10.2f} {ratio:8.3f}")

    # 12) Statistical validation
    print("\n=== STATISTICAL VALIDATION ===")

    # Chi-square test for genre distribution

    # Get common genres (present in both datasets with minimum counts)
    common_genres = set(main_genre_counts.keys()) & set(sub_genre_counts.keys())
    common_genres = [g for g in common_genres if main_genre_counts[g] >= 10 and sub_genre_counts[g] >= 5]

    if len(common_genres) >= 5:  # Need at least 5 categories for chi-square
        main_freq = [main_genre_counts[g] for g in common_genres]
        sub_freq = [sub_genre_counts[g] for g in common_genres]

        # Normalize to same total
        main_total = sum(main_freq)
        sub_total = sum(sub_freq)
        main_norm = [f / main_total for f in main_freq]
        sub_norm = [f / sub_total for f in sub_freq]

        try:
            chi2_stat, p_value = chisquare(sub_norm, main_norm)
            print(f"Chi-square test for genre distribution: χ²={chi2_stat:.3f} p={p_value:.4f}")
            if p_value < 0.05:
                print("⚠️  WARNING: Genre distribution differs significantly from main dataset (p < 0.05)")
            else:
                print("✅ Genre distribution is statistically similar to main dataset (p >= 0.05)")
        except:
            print("Could not perform chi-square test (insufficient data)")
    else:
        print("Insufficient genre overlap for statistical testing")

    # 13) Engagement quality check
    print("\n=== ENGAGEMENT QUALITY VALIDATION ===")

    # Compare engagement metrics by tier
    for t in ['thrash','mid','top']:
        main_tier = meta[meta['pop_tier']==t]
        sub_tier = sample_df[sample_df['pop_tier']==t]

        print(f"\nTier {t.upper()} engagement comparison:")
        print(f"  Main - Median ratings: {main_tier['ratings_count_sum'].median():.0f}, reviews: {main_tier['text_reviews_count_sum'].median():.0f}")
        print(f"  Sub  - Median ratings: {sub_tier['ratings_count_sum'].median():.0f}, reviews: {sub_tier['text_reviews_count_sum'].median():.0f}")

        # Check if sub has higher engagement than main (as intended by sampling)
        if sub_tier['ratings_count_sum'].median() > main_tier['ratings_count_sum'].median():
            print("  ✅ Sub has higher engagement (ratings) than main dataset")
        else:
            print("  ⚠️  Sub has lower engagement (ratings) than main dataset")
        if sub_tier['text_reviews_count_sum'].median() > main_tier['text_reviews_count_sum'].median():
            print("  ✅ Sub has higher engagement (reviews) than main dataset")
        else:
            print("  ⚠️  Sub has lower engagement (reviews) than main dataset")
    # Overall engagement summary
    print("\nOverall engagement summary:")
    main_medians = meta.groupby('pop_tier')[['ratings_count_sum','text_reviews_count_sum']].median()
    sub_medians = sample_df.groupby('pop_tier')[['ratings_count_sum','text_reviews_count_sum']].median()

    print("Main dataset medians:")
    print(main_medians)
    print("\nSub dataset medians:")
    print(sub_medians)

    print(f"\n✅ Enhanced validation completed: {output_csv_path}")
    return sample_df

if __name__ == "__main__":
    # Default paths
    input_path = os.path.join(project_root, "data", "processed", "romance_books_main_final.csv")
    output_path = os.path.join(project_root, "data", "processed", "romance_subdataset_6000.csv")
    
    # Allow command line overrides
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    sample_df = create_subdataset_6000(input_path, output_path)
