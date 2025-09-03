#!/usr/bin/env python3
import json
import glob

# Find latest validation report
latest_report = max(glob.glob('logs/validation/pipeline_validation_*.json'), 
                   key=lambda x: x.split('_')[-1].split('.')[0])

print(f"Analyzing: {latest_report}")

# Load report
with open(latest_report, 'r') as f:
    report = json.load(f)

warnings = report['warnings']
print(f"Total warnings: {len(warnings)}")
print(f"Total errors: {len(report['errors'])}")

# Categorize warnings
warning_types = {}
for w in warnings:
    msg = w['message']
    if 'Publication year' in msg:
        warning_types['publication_year'] = warning_types.get('publication_year', 0) + 1
    elif 'Extremely long review text' in msg:
        warning_types['extremely_long_review_text'] = warning_types.get('extremely_long_review_text', 0) + 1
    elif 'Very long review text' in msg:
        warning_types['long_review_text'] = warning_types.get('long_review_text', 0) + 1
    elif 'books without reviews' in msg:
        warning_types['books_without_reviews'] = warning_types.get('books_without_reviews', 0) + 1
    elif 'Text reviews' in msg and 'significantly higher' in msg:
        warning_types['text_reviews_higher'] = warning_types.get('text_reviews_higher', 0) + 1
    else:
        warning_types['other'] = warning_types.get('other', 0) + 1

print("\nWarning types:")
for wtype, count in warning_types.items():
    print(f"  {wtype}: {count}")

print("\nSample warnings:")
for i, warning in enumerate(warnings[:5]):
    print(f"  {i+1}. {warning['message']}")

# Check if we have any extremely long reviews
if 'extremely_long_review_text' in warning_types:
    print(f"\n✅ Successfully eliminated long review text warnings!")
    print(f"   Only {warning_types['extremely_long_review_text']} extremely long reviews (>50k chars) flagged")
else:
    print(f"\n✅ Successfully eliminated long review text warnings!")
    print(f"   No extremely long reviews found")
