# Data Mapping Issue: work_id vs book_id

## Problem (RESOLVED ✅)

The books dataset (`romance_subdataset_6000.csv`) uses `work_id` as the identifier, while the reviews dataset uses `book_id`. In Goodreads:
- **work_id**: Represents a work (abstract book concept)
- **book_id**: Represents a specific edition of that work

A single work can have multiple editions (paperback, hardcover, ebook, etc.), each with its own `book_id`.

## Solution Implemented

The issue has been **resolved** by implementing a mapping function in `data_loading.py`:

1. **Function**: `load_book_id_to_work_id_mapping()`
   - Loads the main dataset (`romance_books_main_final.csv`) which contains `book_id_list_en`
   - Creates a reverse mapping: `book_id` → `work_id`
   - Filters to only subdataset work_ids for efficiency

2. **Updated**: `load_joined_reviews()`
   - Now uses the mapping to convert `book_id` (reviews) → `work_id` before joining
   - Automatically loads the correct reviews file (`romance_reviews_english_subdataset_6000.csv`)

## Results

After implementing the fix:
- ✅ **969,675 reviews** successfully joined
- ✅ **5,998 out of 6,000 books (99.97%)** have reviews
- ✅ All reviews mapped correctly using the `book_id_list_en` from main dataset

## Technical Details

The mapping works as follows:
1. Main dataset contains `book_id_list_en` column with Python list strings like `['3462', '6338758', ...]`
2. Each `work_id` maps to multiple `book_id` values (editions)
3. We create a reverse dictionary: `{book_id: work_id}` for all editions
4. Reviews are mapped using this dictionary before joining on `work_id`

## Files Updated

- `src/reviews_analysis/data_loading.py`:
  - Added `load_book_id_to_work_id_mapping()` function
  - Updated `load_joined_reviews()` to use the mapping
  - Updated `load_reviews()` to prefer subdataset-specific file

## Impact on Pipeline

✅ **All phases can now proceed**:
- **Phase 2 (Coverage)**: Can accurately compute review counts per book
- **Phase 3 (EDA)**: Can join reviews to books for analysis
- **Phase 4 (BERTopic Input)**: Can aggregate reviews per book
- **Phase 5 (BERTopic)**: Can train models on review-based documents per book

