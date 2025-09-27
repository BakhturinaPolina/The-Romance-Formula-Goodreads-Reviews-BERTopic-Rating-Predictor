"""
Sub-dataset Sampling: Create 6,000 book corpus (2,000 per tier)
Based on the research methodology for balanced representation across popularity tiers.
"""

import pandas as pd
import numpy as np
import os
import sys

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
    meta['series_flag'] = (meta['series_id'].astype(str) != 'stand_alone').astype(int)
    meta['pages_bin']   = pd.qcut(meta['num_pages_median'], q=4, labels=['Q1','Q2','Q3','Q4'])

    def genre_group(s: str):
        s = str(s).lower()
        if 'paranormal' in s:      return 'paranormal'
        if 'historical fiction' in s or 'historical' in s: return 'historical'
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

    # 10) Basic representativeness check (within tiers)
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

    # Engagement sanity check by tier (medians)
    print("\nEngagement medians by tier in SAMPLE:")
    print(sample_df.groupby('pop_tier')[['ratings_count_sum','text_reviews_count_sum']].median())
    
    print(f"\n✅ Sub-dataset created successfully: {output_csv_path}")
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
