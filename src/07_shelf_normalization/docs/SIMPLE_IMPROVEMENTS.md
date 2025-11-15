# Simple Shelf Cleaning Improvements for Topic Modeling

## Goal
Make shelf names cleaner and more useful for topic modeling of novel content and reader reviews, using the **simplest possible approaches**.

## Current State
The pipeline already implements:
- ✅ Canonicalization (case, separators, whitespace, diacritics)
- ✅ Non-content filtering (reading status, formats, dates)
- ✅ Alias detection (complex similarity matching)
- ✅ Segmentation (CamelCase/concat splitting)

## Easiest High-Impact Improvements

### 1. **Plural/Singular Normalization** (EASIEST - High Impact)
**Problem**: "romance" vs "romances", "fantasy" vs "fantasies" are treated as different shelves.

**Simple Solution**: 
- Use simple rules: remove trailing 's'/'es' for common patterns
- Only apply to shelves that already exist in both forms
- Keep both if one is significantly more common

**Implementation**: Add to `canon_key()` function or post-process canonical mapping
```python
def normalize_plural(shelf: str, shelf_counts: Counter) -> str:
    """Simple plural normalization - only if both forms exist."""
    if len(shelf) <= 3:
        return shelf
    
    # Try singular form
    if shelf.endswith('ies'):
        singular = shelf[:-3] + 'y'
    elif shelf.endswith('es') and len(shelf) > 4:
        singular = shelf[:-2]
    elif shelf.endswith('s') and len(shelf) > 2:
        singular = shelf[:-1]
    else:
        return shelf
    
    # Use singular if it exists and is more common
    if singular in shelf_counts and shelf_counts[singular] >= shelf_counts[shelf]:
        return singular
    return shelf
```

**Impact**: Medium-High - Reduces duplicate thematic categories

---

### 2. **Remove Generic Stopwords** (EASY - Medium Impact)
**Problem**: Shelves like "romance-books", "fantasy-novels", "fiction" add noise without thematic value.

**Simple Solution**: 
- Remove common generic words: "book", "books", "novel", "novels", "fiction", "story", "stories"
- Only remove if shelf becomes non-empty after removal
- Keep standalone generic words (they might be meaningful)

**Implementation**: Post-process after canonicalization
```python
GENERIC_STOPWORDS = {'book', 'books', 'novel', 'novels', 'fiction', 'story', 'stories', 'read', 'reading'}

def remove_generic_stopwords(shelf: str) -> str:
    """Remove generic words that don't add thematic value."""
    words = shelf.split()
    filtered = [w for w in words if w not in GENERIC_STOPWORDS]
    result = ' '.join(filtered).strip()
    return result if result else shelf  # Keep original if becomes empty
```

**Impact**: Medium - Reduces noise, focuses on thematic content

---

### 3. **Frequency-Based Filtering** (EASY - Medium Impact)
**Problem**: Very rare shelves (1-2 occurrences) are likely typos, personal tags, or noise.

**Simple Solution**: 
- Filter out shelves that appear < N times (e.g., < 3-5)
- This removes personal tags and typos
- Keep all shelves above threshold

**Implementation**: Already partially done with `--n-top-shelves`, but can be more explicit
```python
MIN_SHELF_FREQUENCY = 3  # Only keep shelves appearing at least 3 times

def filter_rare_shelves(shelf_counts: Counter, min_freq: int = 3) -> Counter:
    """Remove shelves that appear too rarely (likely typos/personal tags)."""
    return Counter({s: c for s, c in shelf_counts.items() if c >= min_freq})
```

**Impact**: Medium - Removes noise, keeps meaningful thematic tags

---

### 4. **Simple Exact Duplicate Merging** (EASIEST - Low-Medium Impact)
**Problem**: After canonicalization, some exact duplicates might still exist due to normalization edge cases.

**Simple Solution**: 
- Group canonical shelves by exact string match
- Merge to most frequent variant
- This should already happen, but can be made explicit

**Implementation**: Already handled by canonicalization, but can add explicit deduplication step

**Impact**: Low-Medium - Ensures consistency

---

### 5. **Common Suffix/Prefix Patterns** (EASY - Low Impact)
**Problem**: "romance-books", "romance-novels" should just be "romance".

**Simple Solution**: 
- Remove common suffixes: "-books", "-novels", "-fiction"
- Remove common prefixes: "books-", "novels-"
- Only if the remaining part is a valid shelf

**Implementation**: 
```python
COMMON_SUFFIXES = ['-books', '-novels', '-fiction', '-story', '-stories']
COMMON_PREFIXES = ['books-', 'novels-', 'fiction-']

def clean_suffixes_prefixes(shelf: str) -> str:
    """Remove common generic suffixes/prefixes."""
    original = shelf
    for suffix in COMMON_SUFFIXES:
        if shelf.endswith(suffix):
            shelf = shelf[:-len(suffix)].strip()
            break
    for prefix in COMMON_PREFIXES:
        if shelf.startswith(prefix):
            shelf = shelf[len(prefix):].strip()
            break
    return shelf if shelf else original
```

**Impact**: Low-Medium - Reduces redundancy

---

## Recommended Implementation Order

1. **Start with #1 (Plural/Singular)** - Easiest, high impact
2. **Add #2 (Stopwords)** - Easy, medium impact  
3. **Add #3 (Frequency filtering)** - Easy, medium impact
4. **Consider #5 (Suffixes/Prefixes)** - Easy, low-medium impact
5. **Skip #4** - Already handled

## Implementation Approach

### Option A: Add to Existing Pipeline (Recommended)
- Modify `canon_key()` function to include plural normalization
- Add post-processing step after canonicalization for stopwords/suffixes
- Add frequency filter before alias detection

### Option B: Create Simple Preprocessing Script
- Standalone script that applies simple rules
- Can be run before or after main normalization
- Easier to test and iterate

## Testing Strategy

1. **Sample Analysis**: Take 1000 random shelves, apply each rule, inspect results
2. **Frequency Check**: Count how many shelves are affected by each rule
3. **Quality Check**: Manual inspection of 50-100 transformed shelves
4. **Impact Measurement**: Compare unique shelf count before/after

## Expected Outcomes

- **Plural normalization**: ~5-10% reduction in unique shelves
- **Stopword removal**: ~10-15% reduction in noise
- **Frequency filtering**: ~20-30% reduction (depends on threshold)
- **Combined**: ~30-40% reduction in unique shelves, better thematic focus

## Notes

- Keep it simple - these are heuristics, not perfect solutions
- Focus on high-frequency shelves (top 10k-50k) for topic modeling
- Document what gets filtered/merged for transparency
- All changes should be reversible (keep original → canonical mapping)

