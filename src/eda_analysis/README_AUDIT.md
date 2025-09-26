# Audit & Heavy-Tails Pack

**Generated:** 2025-01-09

This pack contains a ready-to-run Jupyter notebook and helper files for schema checks,
heavy-tail analysis (Clauset–Shalizi–Newman, 2009), overdispersion tests (Dean–Lawless; Cameron–Trivedi),
and edge case analysis.

## Files
- `01_data_audit_and_heavytails.ipynb` — the main notebook
- `requirements_audit.txt` — suggested Python deps

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements_audit.txt
   ```
2. Set the CSV path (optional):
   ```bash
   export GOODREADS_CSV=/path/to/romance_books_main_final.csv
   ```
3. Run the notebook and execute cells top to bottom.
4. Artifacts will be written to `./audit_artifacts/`.

## Notes
- The CSN step uses the `powerlaw` package (Alstott et al., 2014).
- Overdispersion tests use `statsmodels` (GLM Poisson) and implement
  a Dean–Lawless Pearson-chi² z-test and the Cameron–Trivedi auxiliary OLS.
- Plots use matplotlib only.

## Outputs
- `audit_artifacts/schema_report.json` — column presence & dtype summary
- `audit_artifacts/dtype_summary.csv` — dtypes
- `audit_artifacts/parsed_preview.csv` — first 50 rows after parsing
- `audit_artifacts/ccdf_*.png`, `audit_artifacts/hist_*.png` — heavy-tail visuals
- `audit_artifacts/csn_powerlaw_reports.json` — (if `powerlaw` installed) CSN fits
- `audit_artifacts/overdispersion_tests.json` — Dean–Lawless & Cameron–Trivedi test results
- `audit_artifacts/edge_case_analysis.json` — zero analysis and recommendations

## Key Features
1. **Schema Validation**: Ensures all 19 expected columns are present with correct dtypes
2. **List Parsing**: Robust parsing of list-like strings in `book_id_list_en`, `genres_str`, `shelves_str`
3. **Heavy-Tail Analysis**: Implements CSN (2009) discrete power-law fitting with model comparisons
4. **Overdispersion Tests**: Formal statistical tests to detect Poisson violations
5. **Edge Case Detection**: Identifies zero-inflation and other data quality issues

## References
- Clauset, A., Shalizi, C. R., & Newman, M. E. (2009). Power-law distributions in empirical data. SIAM review, 51(4), 661-703.
- Alstott, J., Bullmore, E., & Plenz, D. (2014). powerlaw: a Python package for analysis of heavy-tailed distributions. PloS one, 9(1), e85777.
- Dean, C., & Lawless, J. F. (1989). Tests for detecting overdispersion in Poisson regression models. Journal of the American Statistical Association, 84(406), 467-472.
- Cameron, A. C., & Trivedi, P. K. (1990). Regression-based tests for overdispersion in the Poisson model. Journal of econometrics, 46(3), 347-364.