# Notebook Protocol (Cursor + Jupyter)

## Purpose
Make our notebooks reproducible, reviewable, and safe for large Goodreads JSON files, while using Cursor effectively.

## Applies to
All notebooks in `notebooks/`.

## Core Principle
One cell at a time: **Plan → Run → Log → Reflect → Decide**.

---

## 0) Tools & Modes

### Run cells in Jupyter notebooks (.ipynb) inside Cursor
Use Cursor's:

- **Agent/Chat** for help writing or explaining code and summarizing outputs
- **Inline Edit** only on selected snippets—never refactor the whole notebook at once
- **Provide precise context** in prompts using `@file`, `@code`, `@folder`, and `@Git` (staged diffs) so the AI focuses on exactly what matters
- **Rules live in `.cursor/rules/`** (we keep them short, focused, and project-scoped). Create or view via New Cursor Rule / Settings → Rules

### Model Choice
- Start with **Auto**; switch to a stronger "thinking/reasoning" model for planning/debugging or to a faster coding model for light edits
- Note any switch in the log

---

## 1) Naming & Location

### Keep notebooks in `notebooks/` with numbered prefixes:
- `01_explore_goodreads_romance_jsons.ipynb` (schema/fields/quality)
- `02_define_quality_filters.ipynb`
- `03_balanced_sampling_by_subgenre.ipynb`
- `04_build_clean_metadata_reviews.ipynb`

### Use clear, descriptive titles inside the first markdown cell (goal, scope, inputs, outputs).

---

## 2) Data Handling Guarantees

- **Raw inputs** are read-only in `data/raw/goodreads_romance/`
- **Interim scratch outputs** go to `data/interim/`
- **Final cleaned subsets** go to `data/processed/`
- **Artifacts** (schemas, samples) go to `artifacts/` and logs to `logs/`

### Respect academic-use restrictions (UCSD Goodreads Book Graph):
- Scraped public shelves, anonymized IDs, no redistribution, non-commercial only
- Source: [cseweb.ucsd.edu](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html#datasets)

---

## 3) The One-Cell Workflow (per cell)

### A. Plan (top of the cell as a comment)
- What will this cell do?
- What files will it read/write?
- What you expect to see (metrics, counts, schema keys)

### B. Run
- Execute the single cell
- Keep operations streaming/chunked for large JSON (never load GBs into RAM)

### C. Log
- Print start/stop timestamps, input paths, sample sizes, and row counts affected
- On writes: print the output path, shape, and a content hash or checksum (if feasible)
- Append a human-readable line to a session log in `logs/<topic>/<YYYYMMDD_HHMM>_<short>.log`

### D. Reflect (markdown cell right after)
3–6 bullets:
- What ran and where it wrote
- Key stats/metrics (shape, null %, sample field names)
- Anomalies/edge cases noticed
- Decision: proceed / adjust / roll back
- Next step proposal (the specific next cell)

**Cursor tip**: When asking the Agent for help with reflection or next steps, pin the current notebook and any relevant config/docs with `@file` so the AI uses the right context.

---

## 4) Error Protocol (observe → hypothesize → test)

When a cell errors:

1. **Capture the full traceback** in the log (don't truncate)
2. **Brainstorm 5–7 plausible causes** (data shape, encoding, path, schema drift, memory, dtype, bad regex, etc.)
3. **Narrow to 1–2 likely causes** and design lightweight instrumentation (extra prints, schema peeks, row counts) to validate
4. **Add only logging first**; rerun to confirm the cause
5. **Only then implement the minimal code fix**, and log the result
6. **Put this checklist (briefly) in the reflection markdown** after the failing cell

---

## 5) Reproducibility & Randomness

- **Seed all random operations** (sampling, shuffling). Record the seed value in both cell output and the log
- **Deterministic I/O**: When sampling from large files, log the filter criteria, the sampling fraction/size, and the seed, and persist the sample as a small artifact (`artifacts/samples/…`)

---

## 6) Performance & Safety for Large JSON

- **Prefer streaming readers and chunked processing** (we'll choose exact tools later)
- **Avoid wide `print()` of entire records**; print schemas, keys, counts, head/tail only
- **For schema discovery**, persist a compact schema summary per file (JSON with field names, value examples, null %, dtype inference) to `artifacts/schemas/`
- **If a step would materialize millions of rows**, first do a 1k–10k sample and review
- **When asking Cursor to suggest approaches**, limit the context to the specific file's header/sample and the current task—don't attach the whole repo

---

## 7) Inputs, Outputs, and Contracts

Each notebook must declare:

- **Inputs**: explicit paths, expected minimal fields (e.g., `book_id`, `work_id`, `author_id`), and size order of magnitude
- **Outputs**: target path, format (CSV/Parquet), expected row count range, and column contract (names, dtypes, allowed values)
- **Failure policy**: what to do if a required field is missing (skip file? flag and stop?)

---

## 8) Subgenre & Balancing (defer decisions to config)

- **Subgenre classification follows `config/` and `docs/` files**; notebooks must not hard-code taxonomy
- **Precedence**: popular shelves signal → book_genres_initial fallback; ties resolved by policy
- **Balancing policy** (target per subgenre, one book/author/subgenre, decade stratification) lives in config; notebooks read and apply it
- **Keep the mapping transparent** in `docs/SUBGENRE_KEYWORDS.md` (human) and `config/subgenre_keywords.yml` (machine). Attach those in Cursor prompts with `@file` when discussing classification

---

## 9) Documentation & Context Discipline (with Cursor)

- **Before you ask the Agent for code or summaries**, attach the precise files via `@file` (e.g., the current notebook, a small sample, and the relevant config page)
- **Keep rule files short and focused**; store them in `.cursor/rules/` and toggle as needed
- **If external docs are required** (e.g., pandas API), attach only the relevant page/section via `@Web` or `@Docs`
- **Use Auto model by default**; document any manual model switches and why (e.g., deep planning vs. quick edit)

---

## 10) Logging Format

**Path**: `logs/<topic>/<YYYYMMDD_HHMM>_<short>.log` (e.g., `logs/exploration/20250818_1210_schema_scan.log`)

- **Header** (per run): timestamp, notebook name, git branch/commit (if clean), data paths, environment (Python, OS)
- **Body**: step name, input sizes, row counts, field inventory (top-N), null %, timing, warnings/errors, outputs written with shapes and hashes
- **Footer**: decision & next step

---

## 11) Git Hygiene (light but consistent)

After a meaningful change (new cell with stable behavior, adjusted config, or updated docs), stage and commit:

- **Message style**: `feat: initial schema inventory for books_romance` or `docs: add subgenre mapping rationale`
- **Ask yourself**: Should README, DATA_DICTIONARY, or configs be updated? If yes, do it in the same PR
- **Use descriptive branches**: `feat/explore-schemas`, `data/sampling-policy`, `fix/review-language-detection`
- **In Cursor, you can draft commit messages with the AI**—attach `@Git` (staged diff) so the message is accurate

---

## Quick Reference

### Cell Template
```python
# PLAN: [What this cell does, files read/written, expected output]
# TIMESTAMP: [Start time]

# [Code here]

# LOG: [End time, metrics, outputs]
```

### Reflection Template
```markdown
## Cell Reflection

- **What ran**: [Brief description]
- **Key metrics**: [Shape, counts, quality stats]
- **Anomalies**: [Any issues noticed]
- **Decision**: [proceed/adjust/rollback]
- **Next step**: [Specific next cell]
```

### Error Analysis Template
```markdown
## Error Analysis

**Traceback**: [Full error message]

**Possible causes**:
1. [Cause 1]
2. [Cause 2]
3. [Cause 3]

**Most likely**: [Top 1-2 causes]

**Validation plan**: [How to test the hypothesis]

**Fix applied**: [What was changed]
```

This protocol ensures systematic, reproducible, and safe notebook development while maximizing the effectiveness of Cursor's AI assistance.
