# Memo: Key Considerations for Topic Modeling

**Date:** November 13, 2025

**Critical Points to Remember:**

- **Account for review volume differences**: Middle tier has highest volume (mean 240.1 reviews/book), top tier has 184.2 reviews/book, trash tier has only 60.7 reviews/book. Consider sampling strategies to balance representation.
- **Middle tier = richest corpus**: The high review count provides excellent material for topic extraction.
- **Top tier = quality + volume**: The combination of high ratings and substantial volume may reveal distinct thematic patterns associated with highly-rated romance novels.
- **Quality gradient**: Ratings increase from trash (3.35) → middle (3.89) → top (4.27). Topic modeling may capture lexical and thematic differences correlating with quality.
- **Modeling approach options**:
    - Tier-stratified topic models (separate models per tier)
    - Use tier as metadata variable in unified BERTopic pipeline

**Action Item:** Decide on sampling strategy and whether to model tiers separately or together with tier metadata before running BERTopic.

The similar median review lengths across tiers (63–70 tokens) indicate that length-based filtering can be applied uniformly, while the tier differences in total volume suggest that sampling strategies may be needed to balance representation if modeling across all tiers simultaneously.

