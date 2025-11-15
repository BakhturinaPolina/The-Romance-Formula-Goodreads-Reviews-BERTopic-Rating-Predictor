# Shelf Category Labelling Guide

**Purpose**: Manual labelling guide for creating `shelf_top_labelled.csv` used to train the hybrid shelf classifier.

## Overview

The hybrid classifier uses **13 category labels** to classify shelf tags. This guide helps you:
1. Understand each category
2. Make consistent labelling decisions
3. Create a high-quality training set

## Category Schema

The classifier uses these **13 categories** (in priority order for rule-based assignment):

### Primary Categories (Character/Pairing Focus)

1. **HEROINE_ARCHETYPE** - Characteristics of the female protagonist
2. **HERO_ARCHETYPE** - Characteristics of the male protagonist  
3. **PAIRING_TYPE** - Relationship structure (M/M, F/F, ménage, etc.)

### Secondary Categories (Content Organization)

4. **STATUS_INTENT** - Reading status (to-read, currently-reading, DNF, etc.)
5. **OWNERSHIP_SOURCE** - How book was acquired (owned, library, ARC, etc.)
6. **FORMAT_MEDIUM** - Physical/digital format (ebook, audiobook, paperback, etc.)
7. **GENRE** - Broad genre classification (romance, fantasy, mystery, etc.)
8. **PLOT_THEME** - Story tropes/themes (enemies-to-lovers, second-chance, etc.)
9. **SETTING_WORLD** - Time period or world type (regency, small-town, vampire, etc.)
10. **TONE_CONTENT** - Emotional tone or heat level (erotica, dark, funny, etc.)
11. **EVALUATION** - Personal rating/opinion (5-stars, favorite, loved-it, etc.)
12. **SERIES_STRUCTURE** - Series membership (series, standalone, trilogy, etc.)
13. **ORG_META** - Organizational metadata (year-based, challenges, etc.)

### Fallback

14. **UNKNOWN_OTHER** - Doesn't fit any category (rare, used as fallback)

---

## Category Definitions & Examples

### 1. HEROINE_ARCHETYPE

**Definition**: Describes characteristics, personality, or archetype of the female protagonist.

**Examples**:
- ✅ `virgin-heroine` - Explicitly about heroine's virginity
- ✅ `strong-heroine` - Heroine's strength/personality
- ✅ `kick-ass-heroine` - Heroine's action-oriented nature
- ✅ `abused-heroine` - Heroine's background/trauma
- ✅ `annoying-heroine` - Opinion about heroine
- ❌ `heroine` (generic) - Too vague, likely GENRE or UNKNOWN_OTHER
- ❌ `heroine-romance` - This is GENRE (romance subgenre)

**Edge Cases**:
- If shelf mentions both hero and heroine characteristics, choose the **more specific** one
- If it's about the **relationship** (not individual), use PLOT_THEME or PAIRING_TYPE

---

### 2. HERO_ARCHETYPE

**Definition**: Describes characteristics, personality, or archetype of the male protagonist.

**Examples**:
- ✅ `alpha-male` - Hero's personality type
- ✅ `alpha-hero` - Hero's archetype
- ✅ `tortured-hero` - Hero's emotional state
- ✅ `damaged-hero` - Hero's background
- ✅ `bad-boy` - Hero's personality
- ✅ `billionaire` - Hero's status (when used as character descriptor)
- ✅ `rock-star` - Hero's profession/archetype
- ❌ `hero` (generic) - Too vague, likely GENRE or UNKNOWN_OTHER
- ❌ `hero-romance` - This is GENRE

**Edge Cases**:
- `billionaire` can be HERO_ARCHETYPE (character) or PLOT_THEME (trope)
  - If it's about the **character type**: HERO_ARCHETYPE
  - If it's about the **story trope** (billionaire romance): PLOT_THEME
- `alpha-male` vs `alpha-heroine`: Choose based on which character is described

---

### 3. PAIRING_TYPE

**Definition**: Describes the relationship structure or pairing type (not individual characters).

**Examples**:
- ✅ `m-m` - Male/male pairing
- ✅ `mm` - Male/male pairing (abbreviation)
- ✅ `f-f` - Female/female pairing
- ✅ `mfm` - Multiple partner (male-female-male)
- ✅ `menage` - Multiple partners
- ✅ `polyamory` - Multiple partners
- ✅ `relationship-m-m` - Explicitly about relationship type
- ❌ `m-f` (if generic) - Default pairing, likely UNKNOWN_OTHER unless explicitly tagged
- ❌ `hero-heroine` - This is GENRE (romance), not pairing type

**Edge Cases**:
- If shelf mentions **both** pairing type and another category (e.g., `m-m-contemporary`):
  - **Priority**: PAIRING_TYPE (more specific)
  - The classifier will learn to handle combinations

---

### 4. STATUS_INTENT

**Definition**: Reading status, intention, or progress (not about book content).

**Examples**:
- ✅ `to-read` - Want to read
- ✅ `tbr` - To be read
- ✅ `currently-reading` - Currently reading
- ✅ `read-2015` - Read in specific year
- ✅ `dnf` - Did not finish
- ✅ `abandoned` - Stopped reading
- ✅ `on-hold` - Paused reading
- ✅ `read-again` - Want to reread
- ❌ `read` (generic) - Too vague, could be STATUS_INTENT or EVALUATION
- ❌ `read-it` - This is EVALUATION (opinion)

**Edge Cases**:
- `read-2015` vs `2015-reads`: Both are STATUS_INTENT (year-based reading status)
- `read-again` vs `reread`: Both are STATUS_INTENT

---

### 5. OWNERSHIP_SOURCE

**Definition**: How the book was acquired or owned (not about content).

**Examples**:
- ✅ `owned` - Own the book
- ✅ `library` - From library
- ✅ `books-i-own` - Ownership
- ✅ `netgalley` - Review copy source
- ✅ `arc` - Advance review copy
- ✅ `freebie` - Free book
- ✅ `amazon-freebie` - Free from Amazon
- ✅ `giveaway` - Won in giveaway
- ❌ `own-it` (if about content) - Could be EVALUATION if expressing opinion
- ❌ `free-read` - This is FORMAT_MEDIUM (free ebook)

**Edge Cases**:
- `owned` vs `books-i-own`: Both are OWNERSHIP_SOURCE
- `freebie` vs `free-ebook`: 
  - `freebie` = OWNERSHIP_SOURCE (how acquired)
  - `free-ebook` = FORMAT_MEDIUM (format type)

---

### 6. FORMAT_MEDIUM

**Definition**: Physical or digital format of the book (not content).

**Examples**:
- ✅ `ebook` - Electronic book
- ✅ `kindle` - Kindle format
- ✅ `audiobook` - Audio format
- ✅ `paperback` - Physical format
- ✅ `hardcover` - Physical format
- ✅ `pdf` - Digital format
- ✅ `epub` - Digital format
- ❌ `ebook-romance` - This is GENRE (romance in ebook format, but genre is primary)
- ❌ `kindle-unlimited` - This is OWNERSHIP_SOURCE (subscription service)

**Edge Cases**:
- `kindle` vs `kindle-unlimited`:
  - `kindle` = FORMAT_MEDIUM
  - `kindle-unlimited` = OWNERSHIP_SOURCE (subscription)
- Format + genre combinations: Choose FORMAT_MEDIUM if format is the primary descriptor

---

### 7. GENRE

**Definition**: Broad genre or subgenre classification.

**Examples**:
- ✅ `romance` - Romance genre
- ✅ `fantasy` - Fantasy genre
- ✅ `contemporary-romance` - Romance subgenre
- ✅ `paranormal-romance` - Romance subgenre
- ✅ `historical-romance` - Romance subgenre
- ✅ `mystery` - Mystery genre
- ✅ `suspense` - Suspense genre
- ✅ `sci-fi` - Science fiction
- ✅ `urban-fantasy` - Fantasy subgenre
- ❌ `romance-books` - Redundant, but still GENRE
- ❌ `genre-romance` - Meta tag, but still GENRE

**Edge Cases**:
- Genre + subgenre combinations: Choose GENRE (e.g., `contemporary-romance` = GENRE)
- Genre + format: Choose GENRE (content is primary)
- Genre + heat level: Choose GENRE (unless heat is explicit like `erotica`, then TONE_CONTENT)

---

### 8. PLOT_THEME

**Definition**: Story tropes, plot devices, or thematic elements.

**Examples**:
- ✅ `enemies-to-lovers` - Relationship trope
- ✅ `friends-to-lovers` - Relationship trope
- ✅ `second-chance` - Plot trope
- ✅ `marriage-of-convenience` - Plot trope
- ✅ `love-triangle` - Plot trope
- ✅ `mafia` - Story theme/setting
- ✅ `sports-romance` - Theme (sports)
- ✅ `military-romance` - Theme (military)
- ✅ `pregnancy` - Plot element
- ✅ `kidnapping` - Plot element
- ❌ `mafia-romance` - This is GENRE (mafia romance subgenre)
- ❌ `sports` (generic) - Could be GENRE if it's sports fiction

**Edge Cases**:
- `mafia` vs `mafia-romance`:
  - `mafia` = PLOT_THEME (mafia as theme/trope)
  - `mafia-romance` = GENRE (romance subgenre)
- `sports-romance` vs `sports`:
  - `sports-romance` = PLOT_THEME (sports as theme in romance)
  - `sports` (generic) = GENRE (sports fiction)

---

### 9. SETTING_WORLD

**Definition**: Time period, geographic location, or world type (not genre).

**Examples**:
- ✅ `regency` - Historical period
- ✅ `victorian` - Historical period
- ✅ `small-town` - Geographic setting
- ✅ `england` - Geographic location
- ✅ `christmas` - Temporal setting
- ✅ `college` - Setting type
- ✅ `vampire` - World type (supernatural)
- ✅ `werewolf` - World type (supernatural)
- ✅ `shifter` - World type (supernatural)
- ❌ `historical` - This is GENRE (historical fiction/romance)
- ❌ `paranormal` - This is GENRE (paranormal romance)

**Edge Cases**:
- `regency` vs `regency-romance`:
  - `regency` = SETTING_WORLD (time period)
  - `regency-romance` = GENRE (romance subgenre)
- `vampire` vs `vampire-romance`:
  - `vampire` = SETTING_WORLD (world type)
  - `vampire-romance` = GENRE (romance subgenre)

---

### 10. TONE_CONTENT

**Definition**: Emotional tone, heat level, or content rating.

**Examples**:
- ✅ `erotica` - Heat level
- ✅ `steamy` - Heat level
- ✅ `dark-romance` - Tone
- ✅ `clean-romance` - Heat level
- ✅ `bdsm` - Content type
- ✅ `angst` - Emotional tone
- ✅ `funny` - Tone
- ✅ `tear-jerker` - Emotional tone
- ✅ `intense` - Tone
- ❌ `romance` (generic) - This is GENRE
- ❌ `dark` (generic) - Could be TONE_CONTENT or GENRE (dark romance)

**Edge Cases**:
- `dark-romance` vs `dark`:
  - `dark-romance` = TONE_CONTENT (tone descriptor)
  - `dark` (generic) = TONE_CONTENT (tone)
- `erotica` vs `erotic-romance`:
  - `erotica` = TONE_CONTENT (heat level)
  - `erotic-romance` = GENRE (romance subgenre with heat descriptor)

---

### 11. EVALUATION

**Definition**: Personal rating, opinion, or evaluation of the book.

**Examples**:
- ✅ `5-stars` - Rating
- ✅ `favorite` - Opinion
- ✅ `favorites` - Opinion
- ✅ `loved-it` - Opinion
- ✅ `meh` - Opinion
- ✅ `boring` - Opinion
- ✅ `not-for-me` - Opinion
- ❌ `read` (generic) - This is STATUS_INTENT
- ❌ `to-read` - This is STATUS_INTENT

**Edge Cases**:
- `favorite` vs `favorites`:
  - Both are EVALUATION (opinion)
- `loved-it` vs `loved`:
  - `loved-it` = EVALUATION (opinion)
  - `loved` (generic) = Could be EVALUATION or STATUS_INTENT (past tense of read)

---

### 12. SERIES_STRUCTURE

**Definition**: Series membership, standalone status, or series structure.

**Examples**:
- ✅ `series` - Part of series
- ✅ `standalone` - Not part of series
- ✅ `trilogy` - Series structure
- ✅ `first-in-series` - Series position
- ✅ `complete-series` - Series status
- ❌ `series-romance` - This is GENRE (series romance subgenre)
- ❌ `book-series` - Redundant, but still SERIES_STRUCTURE

**Edge Cases**:
- `series` vs `series-romance`:
  - `series` = SERIES_STRUCTURE (series membership)
  - `series-romance` = GENRE (romance subgenre about series)

---

### 13. ORG_META

**Definition**: Organizational metadata, challenges, or non-content tags.

**Examples**:
- ✅ `2015-reads` - Year-based organization
- ✅ `reading-challenge` - Challenge tag
- ✅ `botb` - Challenge acronym
- ✅ `to-be-released` - Release status
- ✅ `coming-soon` - Release status
- ❌ `read-2015` - This is STATUS_INTENT (reading status with year)
- ❌ `2015` (generic) - Could be ORG_META or STATUS_INTENT

**Edge Cases**:
- `2015-reads` vs `read-2015`:
  - `2015-reads` = ORG_META (organizational tag)
  - `read-2015` = STATUS_INTENT (reading status)
- `reading-challenge` vs `challenge`:
  - `reading-challenge` = ORG_META
  - `challenge` (generic) = Could be ORG_META or UNKNOWN_OTHER

---

### 14. UNKNOWN_OTHER

**Definition**: Doesn't fit any category, ambiguous, or too generic.

**Examples**:
- ✅ `books` - Too generic
- ✅ `fiction` - Too generic
- ✅ `novels` - Too generic
- ✅ `default` - Meta tag
- ✅ `other` - Catch-all
- ❌ `romance` - This is GENRE
- ❌ `to-read` - This is STATUS_INTENT

**Use UNKNOWN_OTHER when**:
- Shelf is too generic to classify
- Shelf doesn't match any category definition
- Shelf is ambiguous and could fit multiple categories equally
- Shelf is a meta/organizational tag that doesn't fit other categories

---

## Labelling Decision Tree

When labelling a shelf, follow this priority order:

1. **Is it about a character archetype?**
   - Heroine characteristics → **HEROINE_ARCHETYPE**
   - Hero characteristics → **HERO_ARCHETYPE**

2. **Is it about relationship/pairing type?**
   - M/M, F/F, ménage, etc. → **PAIRING_TYPE**

3. **Is it about reading status/intent?**
   - To-read, currently-reading, DNF, etc. → **STATUS_INTENT**

4. **Is it about ownership/acquisition?**
   - Owned, library, ARC, etc. → **OWNERSHIP_SOURCE**

5. **Is it about format?**
   - Ebook, audiobook, paperback, etc. → **FORMAT_MEDIUM**

6. **Is it a genre?**
   - Romance, fantasy, mystery, etc. → **GENRE**

7. **Is it a plot trope/theme?**
   - Enemies-to-lovers, second-chance, etc. → **PLOT_THEME**

8. **Is it about setting/world?**
   - Regency, small-town, vampire, etc. → **SETTING_WORLD**

9. **Is it about tone/heat level?**
   - Erotica, dark, funny, etc. → **TONE_CONTENT**

10. **Is it an evaluation/rating?**
    - 5-stars, favorite, loved-it, etc. → **EVALUATION**

11. **Is it about series structure?**
    - Series, standalone, trilogy, etc. → **SERIES_STRUCTURE**

12. **Is it organizational metadata?**
    - Year-based, challenges, etc. → **ORG_META**

13. **Doesn't fit anywhere?**
    - Generic, ambiguous, meta → **UNKNOWN_OTHER**

---

## Creating `shelf_top_labelled.csv`

### File Structure

Your labelling file should have exactly **2 columns**:

```csv
shelf_canon,human_category
to-read,STATUS_INTENT
romance,GENRE
alpha-male,HERO_ARCHETYPE
m-m,PAIRING_TYPE
virgin-heroine,HEROINE_ARCHETYPE
```

### Labelling Strategy

1. **Start with high-frequency shelves** (from `shelf_canonical_test.csv`, sorted by `count`)
2. **Label systematically**: Work through categories one at a time
3. **Focus on ambiguous cases**: Shelves that could fit multiple categories
4. **Aim for balance**: Try to label examples from all 13 categories
5. **Minimum labels per category**: At least 10-20 examples per category for training
6. **Total labels**: Aim for 200-500 labels minimum (more is better)

### Labelling Tips

1. **Use the decision tree** above for consistency
2. **When in doubt, choose the more specific category**
3. **If truly ambiguous, use UNKNOWN_OTHER** (better than wrong label)
4. **Label the most common/representative meaning** of the shelf
5. **Don't overthink**: First impression is often correct
6. **Batch similar shelves**: Label all "hero" shelves together, then all "heroine" shelves, etc.

### Quality Checks

Before using your labels:

1. **Check for typos**: Ensure category names match exactly (case-sensitive)
2. **Check for balance**: Each category should have at least a few examples
3. **Check for consistency**: Similar shelves should have same category
4. **Check for coverage**: All 13 categories should be represented
5. **Validate against rules**: Compare your labels with rule-based categories

---

## Example Labelling Session

Here's an example of how to label the top 50 shelves:

```csv
shelf_canon,human_category
to-read,STATUS_INTENT
romance,GENRE
currently-reading,STATUS_INTENT
kindle,FORMAT_MEDIUM
ebook,FORMAT_MEDIUM
contemporary,GENRE
series,SERIES_STRUCTURE
favorites,EVALUATION
owned,OWNERSHIP_SOURCE
contemporary-romance,GENRE
fiction,GENRE
ebooks,FORMAT_MEDIUM
adult,GENRE
e-book,FORMAT_MEDIUM
books-i-own,OWNERSHIP_SOURCE
dnf,STATUS_INTENT
erotica,TONE_CONTENT
maybe,STATUS_INTENT
wish-list,OWNERSHIP_SOURCE
alpha-male,HERO_ARCHETYPE
...
```

---

## Common Labelling Mistakes to Avoid

1. **Mixing format and genre**: `ebook-romance` should be **GENRE** (content is primary)
2. **Mixing status and evaluation**: `read-it` is **EVALUATION**, not STATUS_INTENT
3. **Too generic**: `books` should be **UNKNOWN_OTHER**, not GENRE
4. **Character vs. trope**: `alpha-male` is **HERO_ARCHETYPE** (character), not PLOT_THEME
5. **Setting vs. genre**: `regency` is **SETTING_WORLD**, but `regency-romance` is **GENRE**
6. **Heat level vs. genre**: `erotica` is **TONE_CONTENT**, but `erotic-romance` is **GENRE**

---

## Next Steps

1. **Create your labelling file**: Start with `shelf_top_labelled.csv`
2. **Label systematically**: Use this guide and the decision tree
3. **Validate labels**: Check for consistency and balance
4. **Run classifier**: Use `hybrid_classifier.py` to train and predict
5. **Review predictions**: Check `hero_heroine_pairing_predictions.csv` for focus categories
6. **Iterate**: Add more labels if predictions are poor

---

**Questions?** Refer back to category definitions or use UNKNOWN_OTHER when truly uncertain.

