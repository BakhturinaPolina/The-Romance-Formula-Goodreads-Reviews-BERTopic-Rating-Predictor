#!/usr/bin/env python3
"""
Goodreads Shelves (Tags) → Normalized Canonicals → BERTopic

Pipeline overview
-----------------
1) Load Goodreads romance dataset (expects columns incl. `popular_shelves` and/or `shelves_str`).
2) Parse raw shelves per book.
3) Heuristically filter "organizational" shelves (TBR/DNF/owned/yearly, formats, etc.).
4) String normalization (unicode fold, casefold, whitespace, hyphens/underscores, punctuation).
5) Acronym expansion for common book-community terms (YA, NA, HEA/HFN, PNR, etc.).
6) OpenRefine-style fingerprints (token fingerprint & n-gram fingerprint) to form key-collision groups.
7) Fuzzy near-duplicate merge within collision buckets (RapidFuzz).
8) Sentence-Transformer embeddings for remaining unique shelves; nearest-neighbor candidate pairs.
9) Book-level co-occurrence vectors for shelves; nearest neighbors by cosine/Jaccard.
10) Merge shelves when BOTH embedding cosine and co-occurrence similarity exceed thresholds.
11) Connected components → canonical clusters; pick canonical label by frequency & centrality.
12) Save:
    - alias→canonical mapping (with counts)
    - per-book normalized shelves list
    - cluster summaries
13) BERTopic on per-book documents (joined normalized shelves, optional description boost).
14) (Optional) FastAPI app factory to expose `.fit`/`.transform`/`.topics` endpoints.

Usage
-----
# install
pip install -U pandas numpy unidecode rapidfuzz symspellpy hdbscan scikit-learn scipy networkx tqdm sentence-transformers bertopic fastapi uvicorn

# run normalization only
python goodreads_shelf_normalization_and_topics.py normalize \
  --csv data/processed/romance_books_main_final.csv \
  --outdir data/processed/step1_normalized \
  --min-shelf-count 3

# run BERTopic on normalized shelves
python goodreads_shelf_normalization_and_topics.py topics \
  --csv data/processed/romance_books_main_final.csv \
  --norm-map data/processed/step1_normalized/shelf_map.csv \
  --outdir data/processed/step2_topics \
  --append-description false

# run all steps (normalize + topics)
python goodreads_shelf_normalization_and_topics.py all --csv data/processed/romance_books_main_final.csv --outdir data/processed/pipeline_run

# serve FastAPI (optional)
uvicorn goodreads_shelf_normalization_and_topics:create_app --host 0.0.0.0 --port 8000

Notes
-----
- This script is designed to be *safe-by-default*: heavy pairwise ops are restricted to candidate sets via bucketing and nearest neighbors.
- You can adjust thresholds with CLI flags; sensible defaults chosen for short tags.
- Works even if only one of `popular_shelves` or `shelves_str` exists.
- All outputs are CSVs to keep the pipeline portable.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from unidecode import unidecode

# Fuzzy matching
from rapidfuzz import fuzz, process

# Embeddings & ML
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Clustering / graphs
import networkx as nx
import hdbscan

# BERTopic
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired

# Optional API
try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except Exception:  # pragma: no cover
    FastAPI = None  # type: ignore
    BaseModel = object  # type: ignore

# ------------------------------
# Utilities
# ------------------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def read_csv(csv_path: str) -> pd.DataFrame:
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Shape: {df.shape}; columns: {list(df.columns)}")
    return df


def safe_literal_eval(x: str):
    try:
        return json.loads(x)
    except Exception:
        try:
            return ast.literal_eval(x)
        except Exception:
            return x


# ------------------------------
# Parsing shelves from dataframe rows
# ------------------------------

def parse_shelves_from_row(row: pd.Series, min_pop_shelf_count: int = 1) -> List[str]:
    """Return a list of shelf strings for a row.

    Priority: `shelves_str` if present and non-empty; else `popular_shelves` names with count>=min_pop_shelf_count.
    """
    shelves: List[str] = []

    # 1) shelves_str: expected like "['romance', 'historical-fiction']" or pipe/CSV
    if 'shelves_str' in row and isinstance(row['shelves_str'], str) and row['shelves_str'].strip():
        raw = row['shelves_str']
        parsed = safe_literal_eval(raw)
        if isinstance(parsed, (list, tuple)):
            shelves = [str(s) for s in parsed]
        else:
            # try splitting
            shelves = re.split(r"\s*[|,]\s*", str(raw).strip("[] "))
            shelves = [s for s in shelves if s]

    # 2) popular_shelves: usually a list of {name, count}
    if (not shelves) and ('popular_shelves' in row) and isinstance(row['popular_shelves'], str) and row['popular_shelves'].strip():
        raw = row['popular_shelves']
        parsed = safe_literal_eval(raw)
        if isinstance(parsed, (list, tuple)):
            for it in parsed:
                if isinstance(it, dict):
                    name = str(it.get('name', '')).strip()
                    cnt = int(it.get('count', 1) or 1)
                    if name and cnt >= min_pop_shelf_count:
                        shelves.append(name)
                else:
                    # fallback when it's list of names
                    name = str(it).strip()
                    if name:
                        shelves.append(name)
        elif isinstance(parsed, dict):
            for name, cnt in parsed.items():
                name = str(name).strip()
                cnt = int(cnt or 1)
                if name and cnt >= min_pop_shelf_count:
                    shelves.append(name)
        else:
            # attempt to split simple textual list
            items = re.findall(r"'([^']+)'|\"([^\"]+)\"", raw)
            for a, b in items:
                name = a or b
                if name:
                    shelves.append(name)

    return [s for s in shelves if isinstance(s, str) and s.strip()]


# ------------------------------
# Tag classification & normalization
# ------------------------------

ORGANIZATIONAL_KEYWORDS = {
    'tbr', 'to-be-read', 'to_read', 'to-read', 'dnf', 'did-not-finish',
    'owned', 'my-books', 'mybook', 'my-bookshelf', 'kindle', 'kobo', 'audible', 'audiobook', 'ebook', 'e-book',
    'library', 'borrowed', 'netgalley', 'arc', 'advance-copy',
    'signed', 'autographed', 'gifted', 'own', 'owned-ebook', 'paperback', 'hardcover', 'audio',
    'wishlist', 'to-buy', 'buddy-read', 'bookclub', 'book-club',
    'reread', 're-read', 'didn-t-finish', 'dnf-shelf',
}

YEAR_PAT = re.compile(r"\b(19|20)\d{2}\b")
READIN_PAT = re.compile(r"\b(read|reads|read-in|readin)[-_ ]?(19|20)\d{2}\b")
COUNT_PAT = re.compile(r"\b(\d{1,4})\b")

# Minimal acronym expansions common in book communities
ACRONYM_MAP = {
    'tbr': 'to-be-read',
    'dnf': 'did-not-finish',
    'ya': 'young-adult',
    'na': 'new-adult',
    'mg': 'middle-grade',
    'hea': 'happy-ending',  # "happily ever after"
    'hfn': 'happy-for-now',
    'pnr': 'paranormal-romance',
    'romcom': 'romantic-comedy',
}

STOP_TOKENS = set('''a an and the of in on for with to a's an's &'''.split())


def normalize_string(s: str) -> str:
    s = unidecode(str(s)).casefold()
    # Replace separators with spaces
    s = re.sub(r"[\/_.,;:+&]", " ", s)
    s = s.replace('-', ' ')  # hyphens to spaces so tokens align
    s = re.sub(r"\s+", " ", s).strip()
    return s


def expand_acronyms(s: str) -> str:
    toks = s.split()
    out = []
    for t in toks:
        out.append(ACRONYM_MAP.get(t, t))
    return " ".join(out)


def is_organizational_tag(s: str) -> bool:
    base = s.replace('-', ' ')
    if any(k in s for k in ORGANIZATIONAL_KEYWORDS):
        return True
    if YEAR_PAT.search(s) or READIN_PAT.search(s):
        return True
    # overly numeric / counts
    if COUNT_PAT.fullmatch(base):
        return True
    return False


# OpenRefine-like fingerprints
WORD_RE = re.compile(r"[a-z0-9]+")


def token_fingerprint(s: str) -> str:
    toks = [t for t in WORD_RE.findall(s) if t not in STOP_TOKENS]
    toks = sorted(set(toks))
    return " ".join(toks)


def ngram_fingerprint(s: str, n: int = 2) -> str:
    """N-gram fingerprint (see OpenRefine)."""
    s2 = re.sub(r"[^a-z0-9]", "", s)
    if not s2:
        return ""
    grams = sorted(set(s2[i:i+n] for i in range(len(s2) - n + 1)))
    return "".join(grams)


# ------------------------------
# Fuzzy near-duplicate merge (within buckets)
# ------------------------------

class DSU:
    def __init__(self, items: Sequence[str]):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


# ------------------------------
# Candidate generation helpers
# ------------------------------

def bucket_by_prefix(items: List[str], k: int = 3) -> Dict[str, List[str]]:
    buckets: Dict[str, List[str]] = defaultdict(list)
    for s in items:
        key = (s + "###")[:k]
        buckets[key].append(s)
    return buckets


# ------------------------------
# Main normalization pipeline
# ------------------------------

@dataclass
class NormalizationConfig:
    min_pop_shelf_count: int = 1
    min_shelf_count_global: int = 2
    fuzzy_threshold: int = 90  # 0..100 RapidFuzz ratio
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    nn_k: int = 15
    sim_threshold_embed: float = 0.75
    sim_threshold_coocc: float = 0.40
    min_cluster_size_hdbscan: int = 50


@dataclass
class TopicConfig:
    language: str = "multilingual"  # or "english"
    min_topic_size: int = 50
    ngram_max_update: int = 3
    use_keybert_repr: bool = True
    reduce_frequent_words: bool = True


class ShelfNormalizer:
    def __init__(self, cfg: NormalizationConfig):
        self.cfg = cfg
        self.embedder = None  # lazy

    # 1) Extract shelves per book
    def extract_shelves(self, df: pd.DataFrame) -> List[List[str]]:
        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="parse shelves"):
            shelves = parse_shelves_from_row(row, self.cfg.min_pop_shelf_count)
            rows.append(shelves)
        return rows

    # 2) Normalize + filter
    def normalize_and_filter(self, shelves_per_book: List[List[str]]) -> Tuple[List[List[str]], Counter]:
        norm_per_book: List[List[str]] = []
        freq = Counter()
        for lst in shelves_per_book:
            out = []
            for s in lst:
                s0 = normalize_string(s)
                s1 = expand_acronyms(s0)
                if not s1 or is_organizational_tag(s1):
                    continue
                out.append(s1)
            out = list(dict.fromkeys(out))  # de-dup within a book
            freq.update(out)
            norm_per_book.append(out)
        # drop globally rare shelves
        keep = {w for w, c in freq.items() if c >= self.cfg.min_shelf_count_global}
        norm_per_book2 = [[w for w in lst if w in keep] for lst in norm_per_book]
        freq2 = Counter(w for lst in norm_per_book2 for w in lst)
        return norm_per_book2, freq2

    # 3) Key-collision + fuzzy merge (aliasing)
    def alias_merge(self, vocab: List[str]) -> Dict[str, str]:
        # fingerprints
        fp_buckets: Dict[str, List[str]] = defaultdict(list)
        for s in vocab:
            key = token_fingerprint(s) + "|" + ngram_fingerprint(s, 2)
            fp_buckets[key].append(s)
        # also weak prefix buckets
        prefix_buckets = bucket_by_prefix(vocab, 3)

        dsu = DSU(vocab)
        def merge_bucket(items: List[str]):
            if len(items) <= 1:
                return
            # cheap pair pruning: compare to top candidates by WRatio
            # build list for process.cdist-like behavior
            for i in range(len(items)):
                a = items[i]
                # find top candidates w.r.t a
                matches = process.extract(a, items, scorer=fuzz.WRatio, limit=10)
                for b, score, _ in matches:
                    if a == b:
                        continue
                    if score >= self.cfg.fuzzy_threshold:
                        dsu.union(a, b)

        for bucket in fp_buckets.values():
            merge_bucket(bucket)
        for bucket in prefix_buckets.values():
            merge_bucket(bucket)

        # finalize mapping
        groups: Dict[str, List[str]] = defaultdict(list)
        for s in vocab:
            groups[dsu.find(s)].append(s)
        # canonical per group: most frequent, tie-break by shortest then lexicographic
        # We'll attach freq later; for now, placeholder canonical = min by (neg frequency placeholder 0, len, s)
        alias2canon = {}
        for root, items in groups.items():
            canon = sorted(items, key=lambda x: (len(x), x))[0]
            for s in items:
                alias2canon[s] = canon
        return alias2canon

    # 4) Embedding + co-occurrence merge (semantic)
    def semantic_merge(self, shelves_per_book: List[List[str]], alias2canon: Dict[str, str], freq: Counter) -> Dict[str, str]:
        # remap shelves to alias canon from previous step
        canon_books = [[alias2canon.get(s, s) for s in lst] for lst in shelves_per_book]
        canon_vocab = sorted(set(w for lst in canon_books for w in lst))
        index = {w: i for i, w in enumerate(canon_vocab)}

        # book x tag sparse matrix
        rows, cols = [], []
        for bi, lst in enumerate(canon_books):
            for w in set(lst):  # presence
                rows.append(bi)
                cols.append(index[w])
        data = np.ones(len(rows), dtype=np.float32)
        X = csr_matrix((data, (rows, cols)), shape=(len(canon_books), len(canon_vocab)))

        # tag vectors are columns -> use NearestNeighbors on transposed
        nn_co = NearestNeighbors(n_neighbors=min(self.cfg.nn_k + 1, X.shape[1]), metric='cosine')
        nn_co.fit(X.T)
        dist_co, idx_co = nn_co.kneighbors(X.T, return_distance=True)

        # embeddings
        if self.embedder is None:
            self.embedder = SentenceTransformer(self.cfg.embed_model)
        emb = self.embedder.encode(canon_vocab, show_progress_bar=True, normalize_embeddings=True)
        nn_emb = NearestNeighbors(n_neighbors=min(self.cfg.nn_k + 1, emb.shape[0]), metric='cosine')
        nn_emb.fit(emb)
        dist_emb, idx_emb = nn_emb.kneighbors(emb, return_distance=True)

        # union-find over canon_vocab
        dsu = DSU(canon_vocab)
        for i, w in enumerate(canon_vocab):
            # co-occurrence similarity = 1 - cosine distance
            for jpos in range(1, idx_co.shape[1]):  # skip self
                j = idx_co[i, jpos]
                sim_co = 1.0 - float(dist_co[i, jpos])
                if sim_co < self.cfg.sim_threshold_coocc:
                    continue
                # embedding neighbor check: ensure j also among top emb neighbors
                # find sim_emb to j
                # quick lookup: find j index in idx_emb row i
                neighs = idx_emb[i]
                if j in neighs:
                    jpos2 = int(np.where(neighs == j)[0][0])
                    sim_emb = 1.0 - float(dist_emb[i, jpos2])
                else:
                    # approximate by computing cosine manually for this pair
                    v1, v2 = emb[i], emb[j]
                    sim_emb = float(np.dot(v1, v2))
                if sim_emb >= self.cfg.sim_threshold_embed:
                    dsu.union(w, canon_vocab[j])

        # build groups
        groups: Dict[str, List[str]] = defaultdict(list)
        for w in canon_vocab:
            groups[dsu.find(w)].append(w)

        # choose canonical = highest freq; tie by shortest then lexicographic
        alias2canon2: Dict[str, str] = {}
        for root, items in groups.items():
            items_sorted = sorted(items, key=lambda x: (-freq.get(x, 0), len(x), x))
            canon = items_sorted[0]
            for it in items:
                alias2canon2[it] = canon
        # merge with previous
        final_map = {k: alias2canon2.get(v, v) for k, v in alias2canon.items()}
        return final_map


# ------------------------------
# BERTopic wrapper
# ------------------------------

class TopicModeler:
    def __init__(self, tcfg: TopicConfig, embed_model: str):
        self.tcfg = tcfg
        self.embedder = SentenceTransformer(embed_model)

    def fit(self, docs: List[str]) -> BERTopic:
        vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=3)
        ctfidf = ClassTfidfTransformer(reduce_frequent_words=self.tcfg.reduce_frequent_words)
        repr_model = KeyBERTInspired() if self.tcfg.use_keybert_repr else None
        model = BERTopic(
            embedding_model=self.embedder,
            vectorizer_model=vectorizer,
            ctfidf_model=ctfidf,
            representation_model=repr_model,
            min_topic_size=self.tcfg.min_topic_size,
            calculate_probabilities=True,
            language=self.tcfg.language,
        )
        topics, probs = model.fit_transform(docs)
        model.update_topics(docs, n_gram_range=(1, self.tcfg.ngram_max_update))
        return model


# ------------------------------
# Orchestration helpers
# ------------------------------

def run_normalization(csv_path: str, outdir: str, cfg: NormalizationConfig) -> Tuple[pd.DataFrame, Dict[str, str]]:
    ensure_dir(outdir)
    df = read_csv(csv_path)

    # parse shelves per book
    normalizer = ShelfNormalizer(cfg)
    shelves_per_book = normalizer.extract_shelves(df)

    # normalize + filter
    shelves_per_book_norm, freq = normalizer.normalize_and_filter(shelves_per_book)

    # Build vocabulary
    vocab = sorted(set(w for lst in shelves_per_book_norm for w in lst))

    # key-collision + fuzzy alias merge
    alias1 = normalizer.alias_merge(vocab)

    # semantic + co-occurrence merge
    alias_final = normalizer.semantic_merge(shelves_per_book_norm, alias1, freq)

    # Apply mapping to per-book shelves
    books_canon = [[alias_final.get(w, w) for w in lst] for lst in shelves_per_book_norm]
    # de-dup within book again
    books_canon = [sorted(set(lst)) for lst in books_canon]

    # Save alias map with counts
    rows = []
    canon_counts = Counter(w for lst in books_canon for w in lst)
    for alias, canon in sorted(alias_final.items()):
        rows.append({
            'alias': alias,
            'canonical': canon,
            'alias_count': int(freq.get(alias, 0)),
            'canonical_count': int(canon_counts.get(canon, 0)),
        })
    map_df = pd.DataFrame(rows)
    map_path = os.path.join(outdir, 'shelf_map.csv')
    map_df.to_csv(map_path, index=False)

    # Save per-book normalized shelves
    out_df = df.copy()
    out_df['normalized_shelves'] = [json.dumps(lst, ensure_ascii=False) for lst in books_canon]
    out_books_path = os.path.join(outdir, 'books_with_normalized_shelves.csv')
    out_df.to_csv(out_books_path, index=False)

    # Cluster summaries via HDBSCAN on embeddings (for reporting)
    embedder = normalizer.embedder or SentenceTransformer(cfg.embed_model)
    vocab_canon = sorted(set(alias_final.values()))
    emb = embedder.encode(vocab_canon, show_progress_bar=True, normalize_embeddings=True)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=cfg.min_cluster_size_hdbscan, metric='euclidean')
    labels = clusterer.fit_predict(emb)
    cluster_df = pd.DataFrame({'tag': vocab_canon, 'cluster': labels})
    cluster_path = os.path.join(outdir, 'tag_clusters_hdbscan.csv')
    cluster_df.to_csv(cluster_path, index=False)

    print(f"Saved alias map → {map_path}")
    print(f"Saved books with normalized shelves → {out_books_path}")
    print(f"Saved HDBSCAN tag clusters → {cluster_path}")

    return out_df, alias_final


def run_topics(csv_path: str, norm_map_path: str, outdir: str, tcfg: TopicConfig, embed_model: str, append_description: bool = False) -> None:
    ensure_dir(outdir)
    df = read_csv(csv_path)
    # Load mapping
    mp = pd.read_csv(norm_map_path)
    alias2canon = dict(zip(mp['alias'].astype(str), mp['canonical'].astype(str)))

    # reconstruct normalized shelves per book
    normalizer = ShelfNormalizer(NormalizationConfig(embed_model=embed_model))
    shelves_per_book = normalizer.extract_shelves(df)
    shelves_per_book_norm, _ = normalizer.normalize_and_filter(shelves_per_book)
    books_canon = [[alias2canon.get(w, w) for w in lst] for lst in shelves_per_book_norm]
    books_canon = [sorted(set(lst)) for lst in books_canon]

    # Build docs: join shelves; optionally append description for weak signals
    docs = []
    for i, lst in enumerate(books_canon):
        text = " ".join(lst)
        if append_description and isinstance(df.get('description')[i], str):
            desc = normalize_string(df['description'][i])
            text = text + " " + desc
        docs.append(text.strip())

    # Fit BERTopic
    tm = TopicModeler(tcfg, embed_model)
    model = tm.fit(docs)

    # Save topic info
    info = model.get_topic_info()
    info.to_csv(os.path.join(outdir, 'topic_info.csv'), index=False)

    # Per-document topics
    topics, probs = model.transform(docs)
    topics_df = pd.DataFrame({
        'work_id': df.get('work_id', pd.Series(range(len(df)))),
        'topic': topics,
        'prob': [float(p) if p is not None else None for p in probs],
    })
    topics_df.to_csv(os.path.join(outdir, 'book_topics.csv'), index=False)

    # Save model (safetensors) for reuse in API
    model.save(os.path.join(outdir, 'bertopic_model'), serialization='safetensors', save_embedding_model=True)
    print(f"Saved BERTopic artifacts → {outdir}")


# ------------------------------
# FastAPI app factory (optional)
# ------------------------------

def create_app(model_dir: str = None):
    if FastAPI is None:
        raise RuntimeError("fastapi is not installed; pip install fastapi uvicorn")
    app = FastAPI(title="Goodreads Shelves BERTopic API")

    # Lazy-load model
    from bertopic import BERTopic
    _state = {'model': None}
    _model_dir = model_dir or os.environ.get('BERTOPIC_MODEL_DIR', './bertopic_model')

    class FitBody(BaseModel):
        docs: List[str]
        ngram_max: int = 3

    class TransformBody(BaseModel):
        docs: List[str]

    @app.on_event("startup")
    def _load():
        try:
            _state['model'] = BERTopic.load(_model_dir)
            print(f"Loaded BERTopic from {_model_dir}")
        except Exception:
            _state['model'] = None
            print("No pre-trained model found; /fit will create one.")

    @app.post('/fit')
    def fit(body: FitBody):
        if _state['model'] is None:
            # Minimal defaults
            embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=3)
            ctfidf = ClassTfidfTransformer(reduce_frequent_words=True)
            repr_model = KeyBERTInspired()
            _state['model'] = BERTopic(embedding_model=embedder, vectorizer_model=vectorizer, ctfidf_model=ctfidf, representation_model=repr_model, min_topic_size=50, calculate_probabilities=True)
        topics, probs = _state['model'].fit_transform(body.docs)
        _state['model'].update_topics(body.docs, n_gram_range=(1, body.ngram_max))
        _state['model'].save(_model_dir, serialization='safetensors', save_embedding_model=True)
        return {"n_docs": len(body.docs), "n_topics": int(len(set([t for t in topics if t != -1])))}

    @app.post('/transform')
    def transform(body: TransformBody):
        if _state['model'] is None:
            return {"error": "Model not loaded. Call /fit first or set BERTOPIC_MODEL_DIR."}
        topics, probs = _state['model'].transform(body.docs)
        return {"topics": [int(t) for t in topics], "probs": [float(p) if p is not None else None for p in probs]}

    @app.get('/topics')
    def topics():
        if _state['model'] is None:
            return {"error": "Model not loaded."}
        info = _state['model'].get_topic_info().to_dict(orient='records')
        return {"topics": info}

    return app


# ------------------------------
# CLI
# ------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Normalize Goodreads shelves and run BERTopic")
    sub = p.add_subparsers(dest='cmd', required=True)

    # common
    def add_norm_args(ap):
        ap.add_argument('--csv', required=True, help='Path to romance_books_main_final.csv')
        ap.add_argument('--outdir', required=True, help='Output directory')
        ap.add_argument('--min-pop-shelf-count', type=int, default=1)
        ap.add_argument('--min-shelf-count', type=int, default=2)
        ap.add_argument('--fuzzy-threshold', type=int, default=90)
        ap.add_argument('--embed-model', default='sentence-transformers/all-MiniLM-L6-v2')
        ap.add_argument('--nn-k', type=int, default=15)
        ap.add_argument('--sim-embed', type=float, default=0.75)
        ap.add_argument('--sim-coocc', type=float, default=0.40)
        ap.add_argument('--hdbscan-min-cluster', type=int, default=50)

    ap_norm = sub.add_parser('normalize')
    add_norm_args(ap_norm)

    ap_topics = sub.add_parser('topics')
    ap_topics.add_argument('--csv', required=True)
    ap_topics.add_argument('--norm-map', required=True, help='CSV produced by normalize step (shelf_map.csv)')
    ap_topics.add_argument('--outdir', required=True)
    ap_topics.add_argument('--embed-model', default='sentence-transformers/all-MiniLM-L6-v2')
    ap_topics.add_argument('--language', default='multilingual')
    ap_topics.add_argument('--min-topic-size', type=int, default=50)
    ap_topics.add_argument('--ngram-max-update', type=int, default=3)
    ap_topics.add_argument('--append-description', type=lambda x: str(x).lower() in {'1','true','yes','y'}, default=False)

    ap_all = sub.add_parser('all')
    add_norm_args(ap_all)
    ap_all.add_argument('--language', default='multilingual')
    ap_all.add_argument('--min-topic-size', type=int, default=50)
    ap_all.add_argument('--ngram-max-update', type=int, default=3)
    ap_all.add_argument('--append-description', type=lambda x: str(x).lower() in {'1','true','yes','y'}, default=False)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_argparser().parse_args(argv)

    if args.cmd == 'normalize':
        cfg = NormalizationConfig(
            min_pop_shelf_count=args.min_pop_shelf_count,
            min_shelf_count_global=args.min_shelf_count,
            fuzzy_threshold=args.fuzzy_threshold,
            embed_model=args.embed_model,
            nn_k=args.nn_k,
            sim_threshold_embed=args.sim_embed,
            sim_threshold_coocc=args.sim_coocc,
            min_cluster_size_hdbscan=args.hdbscan_min_cluster,
        )
        run_normalization(args.csv, args.outdir, cfg)

    elif args.cmd == 'topics':
        tcfg = TopicConfig(
            language=args.language,
            min_topic_size=args.min_topic_size,
            ngram_max_update=args.ngram_max_update,
        )
        run_topics(
            csv_path=args.csv,
            norm_map_path=args.norm_map,
            outdir=args.outdir,
            tcfg=tcfg,
            embed_model=args.embed_model,
            append_description=args.append_description,
        )

    elif args.cmd == 'all':
        cfg = NormalizationConfig(
            min_pop_shelf_count=args.min_pop_shelf_count,
            min_shelf_count_global=args.min_shelf_count,
            fuzzy_threshold=args.fuzzy_threshold,
            embed_model=args.embed_model,
            nn_k=args.nn_k,
            sim_threshold_embed=args.sim_embed,
            sim_threshold_coocc=args.sim_coocc,
            min_cluster_size_hdbscan=args.hdbscan_min_cluster,
        )
        outdir = args.outdir
        step1_dir = os.path.join(outdir, 'step1_normalized')
        df_out, alias_map = run_normalization(args.csv, step1_dir, cfg)

        # Write map to path for topics step
        norm_map_path = os.path.join(step1_dir, 'shelf_map.csv')
        step2_dir = os.path.join(outdir, 'step2_topics')
        tcfg = TopicConfig(language=args.language, min_topic_size=args.min_topic_size, ngram_max_update=args.ngram_max_update)
        run_topics(args.csv, norm_map_path, step2_dir, tcfg, cfg.embed_model, append_description=args.append_description)


if __name__ == '__main__':
    main()
