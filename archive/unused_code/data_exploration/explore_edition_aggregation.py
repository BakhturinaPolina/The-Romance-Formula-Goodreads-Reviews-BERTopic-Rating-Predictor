#!/usr/bin/env python3
import pandas as pd

# Load the data
df = pd.read_csv('data/processed/romance_books_cleaned.csv')

print('Works with multiple editions (ratings_count_sum > ratings_count):')
multi_editions = df[df['ratings_count_sum'] > df['ratings_count']]
print(f'Found {len(multi_editions)} works with multiple editions')

if len(multi_editions) > 0:
    print("\nSample of works with multiple editions:")
    print(multi_editions[['book_id', 'work_id', 'ratings_count', 'ratings_count_sum', 
                         'average_rating', 'average_rating_weighted_mean']].head(3).to_string())

print(f"\nTotal works processed: {len(df)}")
print(f"Works with single editions: {len(df) - len(multi_editions)}")
print(f"Works with multiple editions: {len(multi_editions)}")
