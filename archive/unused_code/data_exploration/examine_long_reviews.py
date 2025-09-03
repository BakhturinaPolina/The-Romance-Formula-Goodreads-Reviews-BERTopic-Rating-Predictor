#!/usr/bin/env python3
import pandas as pd

# Load reviews data
df = pd.read_csv('data/processed/romance_reviews_20250830_002655.csv')

# Find long reviews
long_reviews = df[df['review_text'].str.len() > 10000]

print(f"Found {len(long_reviews)} reviews longer than 10,000 characters")
print(f"Total reviews: {len(df)}")
print(f"Percentage: {len(long_reviews)/len(df)*100:.3f}%")

print("\nSample of long reviews:")
for i, (idx, row) in enumerate(long_reviews.head(3).iterrows()):
    print(f"\nReview {i+1} (Length: {len(row['review_text'])} chars)")
    print(f"Rating: {row['rating']}")
    print(f"Text preview: {row['review_text'][:200]}...")
    print("-" * 50)
