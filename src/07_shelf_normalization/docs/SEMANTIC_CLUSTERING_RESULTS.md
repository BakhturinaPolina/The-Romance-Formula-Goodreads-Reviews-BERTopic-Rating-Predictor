# Simple Semantic Clustering Results

## Overview

Simple keyword-based semantic clustering of shelves for topic modeling. This is the **EASIEST** approach - no ML, no embeddings, just keyword matching.

## Results Summary

**Dataset**: 93,999 cleaned shelves, 4,057,681 total occurrences  
**Clustering Rate**: 39.7% (1,611,658 occurrences clustered)  
**Unclustered**: 60.3% (2,446,023 occurrences)

## Clustering Approach

### Method: Keyword Matching
- **Simple**: No machine learning, no embeddings
- **Fast**: Processes 94k shelves in ~4 minutes
- **Transparent**: Easy to understand and adjust
- **Rule-based**: Pre-defined keyword dictionaries

### Category Checking Order
1. **Non-content** (reading status, format, year, ownership) - should be minimal
2. **Emotions** (funny, sad, sweet, etc.)
3. **Tropes** (forced-marriage, enemies-to-lovers, etc.)
4. **Genre/Heat** (erotica, hot, steamy, etc.)
5. **Genre** (contemporary, historical, paranormal, etc.)
6. **Unclustered** (if no match found)

## Cluster Categories Found

### 1. Emotion Clusters (5 categories, 47,439 occurrences)

| Category | Occurrences | Top Examples |
|----------|-------------|--------------|
| `funny_humorous` | 21,849 | funny, humor, humorous, humour, romantic-comedy |
| `sweet_romantic` | 13,088 | sweet, cute, sweet-romance, feel-good, heartwarming |
| `sad_emotional` | 10,151 | emotional, made-me-cry, tear-jerker, heartbreaking, sad |
| `angry_intense` | 1,361 | intense, passionate, intense-romance, raw-gritty-intense |
| `happy_joyful` | 990 | happy-ending, happy-ever-after, joy |

**Use for topic modeling**: These can indicate emotional tone of reviews/novels.

---

### 2. Trope Clusters (18 categories, 105,148 occurrences)

| Category | Occurrences | Top Examples |
|----------|-------------|--------------|
| `alpha_male` | 29,048 | alpha-male, alpha-males, alpha, alpha-bad-boy, alpha-hero |
| `mafia_crime` | 11,829 | crime, mafia, mystery-crime, mob, crime-mystery |
| `billionaire` | 10,022 | billionaire, rich, wealthy, millionaire |
| `pregnancy` | 8,806 | pregnancy, baby, pregnant, secret-baby |
| `virgin_heroine` | 8,660 | virgin-heroine, virgin, first-time, virgin-hero |
| `second_chance` | 8,246 | second-chance, reunion, reunited |
| `friends_to_lovers` | 7,250 | friends-to-lovers, best-friends, friend-to-lover |
| `small_town` | 3,888 | small-town, smalltown, small-town-romance |
| `age_gap` | 3,077 | age-gap, older-man, younger-woman, age-difference |
| `enemies_to_lovers` | 2,890 | enemies-to-lovers, hate-to-love, enemy-to-lover |
| `love_triangle` | 2,588 | love-triangle, triangle, love-square |
| `fated_mates` | 2,453 | fated-mates, fated, mate, soulmate, destined |
| `forced_arranged_marriage` | 2,178 | marriage-of-convenience, arranged-marriage, moc, forced-marriage |
| `single_parent` | 1,783 | single-parent, single-mom, single-dad, single-mother |
| `reverse_harem` | 1,076 | reverse-harem, rh, why-choose, multiple-partners |
| `amnesia` | 697 | amnesia, memory-loss, forgot |
| `fake_relationship` | 530 | fake-relationship, fake-dating, pretend-relationship |
| `secret_baby` | 127 | secret-baby, secret-child, hidden-baby |

**Use for topic modeling**: These indicate plot tropes/themes that readers care about.

---

### 3. Genre/Heat Clusters (6 categories, 80,941 occurrences)

| Category | Occurrences | Top Examples |
|----------|-------------|--------------|
| `erotica` | 54,223 | erotica, erotic, erotic-romance, romance-erotica |
| `hot_steamy` | 21,215 | steamy, hot, hot-hot-hot, steamy-romance, hot-romance |
| `sweet_clean` | 3,611 | clean-romance, clean, romance-clean, clean-secular-romance |
| `spicy` | 1,156 | spicy, spice, heat, heated |
| `explicit` | 669 | explicit, sexually-explicit, explicit-sex |
| `closed_door` | 67 | fade-to-black, closed-door-romance, closed-door-fade-to-black |

**Use for topic modeling**: These indicate heat level/sexual content, which is important for romance categorization.

---

### 4. Genre Clusters (10 categories, 371,062 occurrences)

| Category | Occurrences | Top Examples |
|----------|-------------|--------------|
| `contemporary` | 103,507 | contemporary, contemporary-romance, romance-contemporary, m-m-contemporary |
| `paranormal` | 82,086 | paranormal, paranormal-romance, supernatural, magic, shifter |
| `suspense_thriller` | 54,767 | suspense, mystery, romantic-suspense, thriller, mystery-suspense |
| `historical` | 40,656 | historical, historical-romance, regency, regency-romance |
| `fantasy` | 29,637 | fantasy, urban-fantasy, fantasy-romance, sci-fi-fantasy |
| `military` | 13,626 | military, military-romance, military-men, soldier |
| `mc_motorcycle` | 13,281 | club, biker, motorcycle-club, mc, mc-biker |
| `sci_fi` | 11,669 | sci-fi, futuristic, sci-fi-romance, space, sf |
| `western` | 11,022 | western, cowboy, western-romance, cowboy-romance |
| `sports` | 10,811 | sports, sports-romance, sport, athlete, sport-romance |

**Use for topic modeling**: These indicate genre/subgenre classification.

---

### 5. Non-Content Clusters (4 categories, 1,007,068 occurrences)

These should ideally be filtered out before topic modeling:

| Category | Occurrences | Examples |
|----------|-------------|----------|
| `reading_status` | 570,845 | to-read, currently-reading, read, finished, dnf |
| `format` | 286,921 | ebook, kindle, audiobook, hardcover, paperback |
| `ownership` | 145,109 | owned, library, borrowed, wishlist, i-own |
| `year` | 4,193 | read-2012, read-2013, 2012-reads |

**Note**: These are non-thematic and should be excluded from topic modeling.

---

## Unclustered Shelves (2,446,023 occurrences, 60.3%)

Top unclustered shelves (by frequency):
1. `romance` (52,600) - Generic, could be added to genre
2. `series` (34,070) - Indicates series membership
3. `e-book` (31,697) - Format (should be non-content)
4. `favorites` (30,404) - Personal organization
5. `adult` (29,567) - Could be genre/heat indicator
6. `fiction` (27,420) - Generic genre
7. `arc` (17,608) - Format (advance review copy)
8. `maybe` (16,211) - Personal organization
9. `wish-list` (14,929) - Personal organization
10. `to-buy` (13,585) - Personal organization

**Analysis**: Many unclustered shelves are:
- Generic terms (`romance`, `fiction`, `adult`)
- Personal organization (`favorites`, `maybe`, `wish-list`)
- Formats that should be filtered (`e-book`, `arc`)
- Series indicators (`series`, `part-of-a-series`)

## Recommendations for Improvement

### 1. Add Generic Genre Keywords
- `romance` → genre_romance
- `fiction` → genre_fiction
- `adult` → genre_adult or genre_heat_adult

### 2. Expand Trope Keywords
Many trope variants are missed. Consider adding:
- `unwilling-wife` → forced_arranged_marriage
- `marriage-of-convenience` variants
- More specific trope patterns

### 3. Filter Non-Content More Aggressively
- `series`, `part-of-a-series` → non-content
- `arc`, `netgalley` → format (non-content)
- `favorites`, `maybe`, `wish-list` → personal organization (non-content)

### 4. Add Compound Pattern Matching
Some shelves have multiple concepts:
- `contemporary-erotica` → both genre and heat
- `historical-alpha-male` → both genre and trope

**Current behavior**: First match wins (genre checked before trope)

### 5. Consider Multi-Label Clustering
Allow shelves to belong to multiple clusters:
- `contemporary-erotica` → [genre_contemporary, genre_heat_erotica]
- `historical-alpha-male` → [genre_historical, trope_alpha_male]

## Usage for Topic Modeling

### Recommended Approach

1. **Filter non-content shelves** (already done in cleaning step)
2. **Use clustered shelves** as features:
   - Emotion clusters → emotional tone indicators
   - Trope clusters → plot theme indicators
   - Genre/Heat clusters → content classification
   - Genre clusters → subgenre classification

3. **Handle unclustered shelves**:
   - Option A: Ignore (focus on clustered 40%)
   - Option B: Add generic keywords to expand coverage
   - Option C: Use as-is (they might still be meaningful)

### Example Usage

```python
# Load clustered shelves
clustered_df = pd.read_csv('shelf_clusters.csv')

# Filter to content-only clusters
content_clusters = clustered_df[
    ~clustered_df['cluster_type'].isin(['noncontent', 'unclustered'])
]

# Group by cluster for analysis
emotion_shelves = content_clusters[content_clusters['cluster_type'] == 'emotion']
trope_shelves = content_clusters[content_clusters['cluster_type'] == 'trope']
genre_heat_shelves = content_clusters[content_clusters['cluster_type'] == 'genre_heat']
genre_shelves = content_clusters[content_clusters['cluster_type'] == 'genre']
```

## Performance

- **Processing time**: ~4 minutes for 94k shelves
- **Memory usage**: Moderate (pandas DataFrame)
- **Scalability**: Should handle 500k+ shelves without issues

## Files Generated

- `shelf_clusters.csv` - Full clustering results with columns:
  - `shelf_canon`: Original shelf name
  - `count`: Frequency
  - `cluster_category`: Specific category (e.g., "funny_humorous")
  - `cluster_name`: Full cluster name (e.g., "emotion_funny_humorous")
  - `cluster_type`: Cluster type (emotion, trope, genre_heat, genre, noncontent, unclustered)

- `clustering_stats.json` - Detailed statistics and examples

## Next Steps

1. ✅ Test clustering on data - **DONE**
2. ⏳ Expand keyword dictionaries based on unclustered patterns
3. ⏳ Add generic genre keywords to increase coverage
4. ⏳ Consider multi-label clustering for compound shelves
5. ⏳ Integrate with topic modeling pipeline

## Advantages of This Approach

1. **Simple**: No ML/embeddings needed
2. **Fast**: Keyword matching is very fast
3. **Transparent**: Easy to see why shelves are clustered
4. **Adjustable**: Easy to add/remove keywords
5. **Interpretable**: Cluster names are meaningful

## Limitations

1. **Coverage**: Only 40% clustered (could be improved with more keywords)
2. **No fuzzy matching**: Exact keyword match required
3. **Single label**: One shelf → one cluster (no multi-label)
4. **Manual keywords**: Requires domain knowledge to build dictionaries

## Alternative Approaches (If Needed)

If coverage needs to be higher:
1. **String similarity**: Use edit distance/Jaccard to group similar shelves
2. **Embeddings**: Use word2vec/fastText to find semantic similarity
3. **Topic modeling**: Cluster shelves using topic modeling itself
4. **Hybrid**: Combine keyword matching + similarity for unclustered

But for now, **keyword matching is the easiest and most interpretable**.

