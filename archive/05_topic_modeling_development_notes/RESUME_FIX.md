# Fix: Resume Optimization Using Old Search Space

## Problem Identified

After fixing the config file to restrict `hdbscan_metric` to `['euclidean', 'l2']`, the optimization was still failing because:

1. **Old result.json files existed** with the previous search space containing `['euclidean', 'manhattan', 'cosine']`
2. **Optimizer resumes from saved files** - When `optimizer.resume_optimization()` is called, it loads the search space from the existing `result.json` file (see `optimizer.py` line 472-473)
3. **Config is ignored on resume** - The new config file's search space is only used when starting a fresh optimization

## Evidence from Logs

The terminal output showed:
- **New config loaded correctly**: `hdbscan__metric: Categorical(2 values)` (lines 328, 392)
- **But saved results had old values**: `'hdbscan__metric': ['Categorical', ['euclidean', 'manhattan', 'cosine'], None]` (lines 496-498, 575-577, etc.)

## Solution Applied

1. **Backed up old result.json files** to `.old_<timestamp>` 
2. **Removed old result.json files** so optimization starts fresh
3. **Restarted optimization** - Now it will create new result.json files with the correct search space

## Files Backed Up

The following old result files were backed up:
- `sentence-transformers/paraphrase-mpnet-base-v2/result.json.old_20251115_002538`
- `sentence-transformers/multi-qa-mpnet-base-cos-v1/result.json.old_20251115_002538`
- `intfloat/e5-base-v2/result.json.old_20251115_002538`
- `thenlper/gte-large/result.json.old_20251115_002539`

## Expected Behavior Now

1. ✅ Optimization starts fresh (no resume from old files)
2. ✅ New search space built from updated config file
3. ✅ `hdbscan__metric` will only have `['euclidean', 'l2']` values
4. ✅ No metric compatibility errors
5. ✅ Training runs should complete successfully

## Verification

To verify the fix worked, check the new result.json files after optimization starts:
```bash
# Check search space in new result files
grep -A 3 "hdbscan__metric" data/interim/octis_reviews/optimization_results/*/result.json

# Should show only: ['euclidean', 'l2']
# NOT: ['euclidean', 'manhattan', 'cosine']
```

## Future Prevention

If config changes are made in the future:
1. **Delete or backup old result.json files** before resuming
2. **Or modify optimizer code** to update search space when resuming (more complex)
3. **Document config changes** that require fresh optimization start

