# Corpus Construction and Statistical Validation

## Dataset Overview

We constructed a research corpus of romance novels from Goodreads metadata, beginning with a full dataset of 52,585 English-language romance novels published between 2000 and 2017. From this full dataset, we selected a stratified sample of 6,000 novels using a multi-tier sampling strategy based on popularity metrics.

### Full Dataset Characteristics

The full dataset comprises 52,585 romance novels from 17,584 unique authors, with 35,469 books belonging to series and 17,116 standalone works (67.5% standalone, 32.5% series). Publication years range from 2000 to 2017, with a mean of 2012.8 (SD = 3.3) and median of 2013. The dataset exhibits substantial variation in reader engagement: average ratings (weighted mean across editions) range from 1.27 to 5.00 (M = 3.91, Mdn = 3.93, SD = 0.35), while ratings counts range from 1 to 1,686,868 (M = 1,749, Mdn = 182). Text review counts range from 0 to 74,298 (M = 143, Mdn = 32). Page counts (median across editions) range from 90 to 980 pages (M = 261.2, Mdn = 256.0, SD = 101.1).

Genre distribution analysis identified 10 unique genre categories after canonicalization. The most frequent genres were romance (100% of books, by definition), fiction (69.9%), paranormal (28.6%), fantasy (28.6%), mystery (26.8%), young adult (15.6%), and historical romance (15.1%).

### Subset Selection Strategy

The 6,000-book research subset was constructed using a stratified sampling approach designed to ensure equal representation across three popularity tiers (2,000 books per tier) while preserving demographic characteristics of the full dataset. Popularity tiers were defined by quartiles of `average_rating_weighted_mean`: thrash (bottom 25%, < 3.71), mid (middle 50%, 3.71–4.15), and top (top 25%, > 4.15). Within each tier, books were prioritized by engagement metrics (ratings count, text reviews count, author ratings count) to maximize data richness for downstream analysis. Stratification preserved distributions of publication year, genre groups, series status, and page length quartiles.

## Statistical Validation

To assess the representativeness of the 6,000-book subset, we performed comprehensive statistical comparisons between the full dataset and subset across key demographic and engagement variables.

### Categorical Variables

**Genre Distribution.** A chi-square test comparing genre frequencies between the full dataset and subset revealed no significant difference (χ² = 0.011, p > 0.999), indicating that the subset maintains the genre composition of the full dataset.

**Publication Year Distribution.** Chi-square test on decade bins (2000s, 2010s) showed no significant difference (χ² = 2.06, p = 0.151, Cramér's V = 0.006). A Mann-Whitney U test on continuous publication year values also found no significant difference (U = 155,650,108, p = 0.087, Cohen's d = -0.026), confirming temporal representativeness.

**Series Status.** Chi-square test comparing proportions of standalone vs. series books found no significant difference (χ² = 1.73, p = 0.188, Cramér's V = 0.005), indicating the subset preserves the series/standalone distribution.

### Continuous Variables

**Average Rating.** Mann-Whitney U test revealed a statistically significant but small-magnitude difference in average ratings between full dataset (M = 3.91, Mdn = 3.93) and subset (M = 3.93, Mdn = 3.98), U = 151,898,978, p < 0.001, Cohen's d = -0.061. This small effect size (d < 0.2) indicates minimal practical difference.

**Engagement Metrics.** As expected given the sampling strategy prioritizing high-engagement books, the subset showed significantly higher engagement metrics. Ratings count sum was significantly higher in the subset (M = 8,888, Mdn = 2,661) compared to the full dataset (M = 1,749, Mdn = 182), U = 48,396,241, p < 0.001, Cohen's d = -0.42 (medium effect). Similarly, text reviews count sum was significantly higher in the subset (M = 683, Mdn = 264) compared to the full dataset (M = 143, Mdn = 32), U = 50,955,036, p < 0.001, Cohen's d = -0.57 (medium-to-large effect). These differences are intentional and reflect the sampling strategy's prioritization of books with rich reader engagement data.

**Page Count.** Mann-Whitney U test found no significant difference in page counts between full dataset (M = 261.2, Mdn = 256.0) and subset (M = 261.7, Mdn = 253.0), U = 157,508,606, p = 0.843, Cohen's d = -0.005, confirming that the subset maintains the length distribution of the full dataset.

### Summary

Statistical validation confirms that the 6,000-book subset maintains representativeness across key demographic variables (genre, publication year, series status, page count) while intentionally prioritizing books with higher reader engagement. The subset preserves the structural characteristics of the full dataset while providing richer data for analysis, making it well-suited for research requiring substantial reader engagement metrics.

## Statistical Test Results Summary

| Variable | Test | Statistic | p-value | Effect Size | Interpretation |
|----------|------|-----------|---------|-------------|----------------|
| Genre distribution | χ² | 0.011 | > 0.999 | — | No significant difference |
| Publication year (decade) | χ² | 2.06 | 0.151 | Cramér's V = 0.006 | No significant difference |
| Publication year (continuous) | Mann-Whitney U | 155,650,108 | 0.087 | Cohen's d = -0.026 | No significant difference |
| Series status | χ² | 1.73 | 0.188 | Cramér's V = 0.005 | No significant difference |
| Average rating | Mann-Whitney U | 151,898,978 | < 0.001 | Cohen's d = -0.061 | Significant, small effect |
| Ratings count | Mann-Whitney U | 48,396,241 | < 0.001 | Cohen's d = -0.42 | Significant, medium effect* |
| Text reviews count | Mann-Whitney U | 50,955,036 | < 0.001 | Cohen's d = -0.57 | Significant, medium-large effect* |
| Page count | Mann-Whitney U | 157,508,606 | 0.843 | Cohen's d = -0.005 | No significant difference |

*Expected differences due to sampling strategy prioritizing high-engagement books.

