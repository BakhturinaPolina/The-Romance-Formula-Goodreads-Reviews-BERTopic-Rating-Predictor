#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Shelf Classifier

Combines rule-based categories with sentence embeddings to classify shelf tags:
- Rule-based category (rule_category) via keyword matching
- Sentence embeddings (all-mpnet-base-v2)
- One-hot(rule_category) + embeddings -> Logistic Regression
- Trained on human-labelled subset, predicts for all shelves

Usage:
    python hybrid_classifier.py \
        --input outputs/test/shelf_canonical_test.csv \
        --labels outputs/test/shelf_top_labelled.csv \
        --output outputs/test/shelves_with_hybrid_categories.csv \
        --model outputs/test/shelf_hybrid_model.pkl \
        --inspect-hero-pairing outputs/test/hero_heroine_pairing_predictions.csv
"""

from __future__ import annotations
import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==========================
# 1. RULE-BASED CATEGORIES
# ==========================

HEROINE_KEYWORDS = [
    "heroine",
    "virgin heroine",
    "abused heroine",
    "strong heroine",
    "kick-ass heroine",
    "kickass heroine",
    "annoying heroine",
]

HERO_KEYWORDS = [
    " hero",                 # space to avoid matching 'heroine'
    "rich hero",
    "tortured hero",
    "damaged hero",
    "alpha male",
    "alpha-male",
    "alpha-hero",
    "bad boy",
    "bad-boy",
    "billionaire",
    "billionaires",
    "rock-star",
    "rockstar",
    "rockstars",
    "rock-stars",
]

PAIRING_KEYWORDS = [
    "m-m", "mm", "male-male", "m m",
    "f-f", "ff", "female-female", "f f",
    "m-f", "mf", "m f",
    "mfm", "mmf",
    "menage", "ménage",
    "multiple-partners", "polyamory",
    "relationship-m-m", "relationship m m",
]

STATUS_KEYWORDS = [
    "to read", "tbr", "own tbr", "own-tbr", "tbr-own", "tbr pile",
    "currently reading",
    "read 201", "read-in-201", "2010-reads", "2011-reads", "2012-reads",
    "2013-reads", "2014-reads", "2015-reads", "2016-reads", "2017-reads",
    "dnf", "did not finish", "didn t finish", "couldn t finish",
    "abandoned", "on hold",
    "waiting-to-be-read", "up next", "next-up", "next-to-read",
    "read soon", "to-read-soon", "queued", "read-again", "reread",
]

OWNERSHIP_KEYWORDS = [
    "owned", "books i own", "my books", "my-owned-books",
    "own-it", "i-own", "have-it", "have but not read",
    "owned not read", "own-but-not-read", "owned-but-unread",
    "freebie", "freebies", "free read", "free-reads", "free-books",
    "amazon-freebie", "free-ebook", "free-ebooks", "free-kindle",
    "library", "library-book", "library-books", "in-library",
    "not-at-library",
    "netgalley", "arc", "edelweiss", "hoopla", "scribd",
    "giveaway", "giveaways", "signed",
]

FORMAT_KEYWORDS = [
    "ebook", "e-book", "e books", "ebooks", "digital", "pdf", "epub", "epubs",
    "kindle", "kindle-", "kindle ", "kindleunlimited", "kobo", "nook",
    "ibooks", "ibook", "ipad",
    "audiobook", "audio-book", "audiobooks", "audio books", "audio ",
    "paperback", "paperbacks",
    "format-kindle",
]

GENRE_KEYWORDS = [
    # base genres
    "romance", "fantasy", "mystery", "thriller", "suspense",
    "crime", "horror", "sci-fi", "science-fiction", "scifi",
    "historical", "womens-fiction", "women s fiction",
    "christian-fiction", "inspirational",
    # audience
    "ya ", " young-adult", " teen", "children", "adult-fiction",
    # romance subs & genre-*
    "paranormal-romance", "romantic-suspense", "romantic-comedy",
    "romance-erotica", "romance-erotic", "dark-romance",
    "urban-fantasy", "paranormal-fantasy", "fantasy-paranormal",
    "sci-fi-fantasy", "fantasy-sci-fi",
    "genre-romance", "genre-contemporary-romance",
    "genre-paranormal", "genre-fantasy",
]

PLOT_KEYWORDS = [
    "friends-to-lovers", "enemies-to-lovers",
    "second-chance", "second-chances", "second-chance-romance",
    "marriage-of-convenience", "love-triangle", "insta-love",
    "opposites-attract", "forbidden-love",
    "pregnancy", "kidnapping", "kidnapped", "abuse", "rape",
    "hurt-comfort", "disability", "revenge", "betrayal",
    "mafia", "secrets", "motorcycle-club", "mc-books", "bikers",
    "sports-romance", "sport", "athlete", "athletes",
    "military-romance", "military-men", "law-enforcement",
    "office-romance", "rock-star", "rockstar", "rockstars", "musician",
]

SETTING_KEYWORDS = [
    "regency", "victorian",
    "historical-romance", "historicals",
    "western", "western-romance",
    "small-town", "small-town-romance",
    "england", "british",
    "christmas", "holiday", "holidays",
    "war",
    "college", "college-romance",
    "high-school",
    # supernatural species as world/setting flavour
    "vampire", "vampires", "vamps",
    "werewolf", "werewolves", "wolves",
    "shifter", "shifters", "shapeshifter", "shapeshifters",
    "dragons", "fae", "angels", "demons", "angels-demons",
    "aliens", "alien", "ghosts",
]

TONE_KEYWORDS = [
    "erotica", "erotic", "smut", "steamy", "steamy-romance",
    "clean-romance", "clean ",
    "dark-romance", "dark-erotica", "dark ",
    "bdsm", "kinky", "kink",
    "angst", "angsty",
    "tear-jerker", "made-me-cry", "heartbreaking",
    "funny", "humor", "humour", "hilarious", "fun",
    "intense", "guilty-pleasure", "guilty-pleasures",
]

EVAL_KEYWORDS = [
    "5-stars", "5-star", "4-stars", "4-star",
    "3-stars", "3-star", "2-stars",
    "4-5-stars", "3-5-stars",
    "favorite", "favourite", "favorites", "favourites",
    "favorite-books", "my-favorites", "my-favorites",
    "all-time-favorites", "faves", "fav",
    "loved-it", "loved",
    "meh", "boring", "not-for-me", "nope", "no-thanks",
]

SERIES_KEYWORDS = [
    "series", "part-of-a-series", "part-of-series",
    "series-books", "series-romance", "series-to-read",
    "series-to-start", "series-sequels", "series-to-finish",
    "book-series",
    "stand-alone", "standalone", "stand-alones", "single-title",
    "trilogy",
    "complete-series", "completed-series",
    "first-in-series", "1st-in-series", "first-in-a-series",
]

ORG_META_KEYWORDS = [
    "2010-reads", "2011-reads", "2012-reads", "2013-reads",
    "2014-reads", "2015-reads", "2016-reads", "2017-reads",
    "books-read-in-201", "2014-books", "2015-books",
    "2016-books", "2017-books",
    "reading-challenge", "2014-reading-challenge",
    "2015-reading-challenge", "2016-reading-challenge",
    "2017-reading-challenge", "botb",
    " release", " releases", "to-be-released", "coming-soon",
    " default", " other ", " history",
]


def normalise_shelf(shelf: str) -> str:
    """Normalize shelf string for keyword matching."""
    # Handle NaN/None values
    if pd.isna(shelf) or not isinstance(shelf, str):
        return ""
    s = shelf.lower()
    s = s.replace("_", " ")
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def contains_any(s: str, keywords: List[str]) -> bool:
    """Check if normalized string contains any keyword."""
    return any(kw in s for kw in keywords)


def assign_category(shelf_canon: str) -> str:
    """
    Rule-based mapping from shelf_canon string to a coarse category label.
    
    Returns one of:
    HEROINE_ARCHETYPE, HERO_ARCHETYPE, PAIRING_TYPE,
    STATUS_INTENT, OWNERSHIP_SOURCE, FORMAT_MEDIUM,
    GENRE, PLOT_THEME, SETTING_WORLD, TONE_CONTENT,
    EVALUATION, SERIES_STRUCTURE, ORG_META, UNKNOWN_OTHER
    """
    # Handle NaN/None values
    if pd.isna(shelf_canon) or not isinstance(shelf_canon, str):
        return "UNKNOWN_OTHER"
    
    s = normalise_shelf(shelf_canon)
    
    # Handle empty strings after normalization
    if not s:
        return "UNKNOWN_OTHER"

    # 1) heroine/hero/pairing FIRST (your main interest)
    if contains_any(s, HEROINE_KEYWORDS):
        return "HEROINE_ARCHETYPE"
    if contains_any(s, HERO_KEYWORDS):
        return "HERO_ARCHETYPE"
    if contains_any(s, PAIRING_KEYWORDS):
        return "PAIRING_TYPE"

    # 2) other conceptual axes, in priority order
    if contains_any(s, STATUS_KEYWORDS):
        return "STATUS_INTENT"
    if contains_any(s, OWNERSHIP_KEYWORDS):
        return "OWNERSHIP_SOURCE"
    if contains_any(s, FORMAT_KEYWORDS):
        return "FORMAT_MEDIUM"
    if contains_any(s, GENRE_KEYWORDS):
        return "GENRE"
    if contains_any(s, PLOT_KEYWORDS):
        return "PLOT_THEME"
    if contains_any(s, SETTING_KEYWORDS):
        return "SETTING_WORLD"
    if contains_any(s, TONE_KEYWORDS):
        return "TONE_CONTENT"
    if contains_any(s, EVAL_KEYWORDS):
        return "EVALUATION"
    if contains_any(s, SERIES_KEYWORDS):
        return "SERIES_STRUCTURE"
    if contains_any(s, ORG_META_KEYWORDS):
        return "ORG_META"

    return "UNKNOWN_OTHER"


# ==========================
# MAIN CLASSIFICATION PIPELINE
# ==========================

def load_data(input_path: Path) -> pd.DataFrame:
    """Load shelf canonical data."""
    logger.info(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Validate required columns
    required_cols = ["shelf_canon"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    logger.info(f"Loaded {len(df):,} shelves")
    return df


def assign_rule_categories(df: pd.DataFrame, checkpoint_path: Optional[Path] = None) -> pd.DataFrame:
    """Assign rule-based categories to all shelves."""
    # Check for checkpoint
    if checkpoint_path and checkpoint_path.exists():
        logger.info(f"Loading rule categories from checkpoint: {checkpoint_path}")
        checkpoint_df = pd.read_csv(checkpoint_path)
        if "rule_category" in checkpoint_df.columns and len(checkpoint_df) == len(df):
            logger.info("Using existing rule categories from checkpoint")
            df = df.copy()
            df["rule_category"] = checkpoint_df["rule_category"].values
            # Log distribution
            rule_dist = df["rule_category"].value_counts()
            logger.info("Rule category distribution:")
            for cat, count in rule_dist.items():
                logger.info(f"  {cat}: {count:,} ({100*count/len(df):.1f}%)")
            return df
    
    logger.info("Assigning rule-based categories...")
    df = df.copy()
    df["rule_category"] = df["shelf_canon"].apply(assign_category)
    
    # Save checkpoint
    if checkpoint_path:
        logger.info(f"Saving rule categories checkpoint to {checkpoint_path}")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        df[["shelf_canon", "rule_category"]].to_csv(checkpoint_path, index=False)
    
    # Log distribution
    rule_dist = df["rule_category"].value_counts()
    logger.info("Rule category distribution:")
    for cat, count in rule_dist.items():
        logger.info(f"  {cat}: {count:,} ({100*count/len(df):.1f}%)")
    
    return df


def compute_embeddings(df: pd.DataFrame, model_name: str = "sentence-transformers/all-mpnet-base-v2", 
                      checkpoint_path: Optional[Path] = None) -> pd.DataFrame:
    """Compute sentence embeddings for all shelves."""
    # Check for checkpoint
    if checkpoint_path and checkpoint_path.exists():
        logger.info(f"Loading embeddings from checkpoint: {checkpoint_path}")
        try:
            checkpoint_data = joblib.load(checkpoint_path)
            checkpoint_embeddings = checkpoint_data["embeddings"]
            checkpoint_shelves = checkpoint_data.get("shelf_canon", None)
            checkpoint_model = checkpoint_data.get("model_name", None)
            
            # Verify checkpoint matches current data
            if len(checkpoint_embeddings) == len(df):
                # Check model name matches (if stored)
                if checkpoint_model and checkpoint_model != model_name:
                    logger.warning(f"Checkpoint model mismatch ({checkpoint_model} vs {model_name}). Recomputing embeddings...")
                else:
                    # If shelves are stored, verify they match (set comparison for order independence)
                    if checkpoint_shelves is not None:
                        current_shelves_set = set(df["shelf_canon"].dropna().astype(str))
                        checkpoint_shelves_set = set(pd.Series(checkpoint_shelves).dropna().astype(str))
                        if current_shelves_set != checkpoint_shelves_set:
                            logger.warning("Checkpoint shelves don't match current data (different shelves). Recomputing embeddings...")
                        else:
                            # Shelves match, use checkpoint (reorder embeddings if needed)
                            logger.info("Using existing embeddings from checkpoint (reordering if needed)")
                            df = df.copy()
                            # Create mapping from shelf to embedding index
                            if len(checkpoint_shelves) == len(df):
                                # Try to match by position first (fast path)
                                if all(cs == sc for cs, sc in zip(checkpoint_shelves, df["shelf_canon"])):
                                    df["embed"] = list(checkpoint_embeddings)
                                else:
                                    # Reorder embeddings to match current dataframe order
                                    shelf_to_idx = {shelf: idx for idx, shelf in enumerate(checkpoint_shelves)}
                                    reordered_embeddings = np.array([
                                        checkpoint_embeddings[shelf_to_idx.get(shelf, 0)]
                                        for shelf in df["shelf_canon"]
                                    ])
                                    df["embed"] = list(reordered_embeddings)
                            else:
                                df["embed"] = list(checkpoint_embeddings)
                            logger.info(f"Loaded {len(checkpoint_embeddings):,} embeddings of dimension {checkpoint_embeddings.shape[1]}")
                            return df
                    else:
                        # No shelf info stored, just use embeddings if size matches
                        logger.info("Using existing embeddings from checkpoint (no shelf validation)")
                        df = df.copy()
                        df["embed"] = list(checkpoint_embeddings)
                        logger.info(f"Loaded {len(checkpoint_embeddings):,} embeddings of dimension {checkpoint_embeddings.shape[1]}")
                        return df
            else:
                logger.warning(f"Checkpoint size mismatch ({len(checkpoint_embeddings)} vs {len(df)}). Recomputing embeddings...")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Recomputing embeddings...")
    
    logger.info(f"Loading embedding model: {model_name}...")
    embedder = SentenceTransformer(model_name)

    logger.info("Computing embeddings (this may take a while)...")
    embeddings = embedder.encode(
        df["shelf_canon"].tolist(),
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=32,  # Process in batches for memory efficiency
    )
    df = df.copy()
    df["embed"] = list(embeddings)
    
    # Save checkpoint
    if checkpoint_path:
        logger.info(f"Saving embeddings checkpoint to {checkpoint_path}")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "embeddings": embeddings,
            "shelf_canon": df["shelf_canon"].values,
            "model_name": model_name,
        }, checkpoint_path)
        logger.info("Embeddings checkpoint saved")
    
    logger.info(f"Computed {len(embeddings):,} embeddings of dimension {embeddings.shape[1]}")
    return df


def prepare_features(df: pd.DataFrame, encoder: Optional[OneHotEncoder] = None) -> Tuple[np.ndarray, OneHotEncoder]:
    """
    Prepare feature matrix: [embeddings | one-hot(rule_category)].
    
    Returns:
        (feature_matrix, encoder)
    """
    logger.info("Preparing features...")
    
    # Stack embeddings
    X_embed = np.vstack(df["embed"].values)
    logger.info(f"Embedding features: {X_embed.shape}")
    
    # One-hot encode rule_category
    if encoder is None:
        # Use sparse_output instead of sparse (scikit-learn 1.2+)
        try:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:
            # Fallback for older scikit-learn versions
            encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        rule_feat = encoder.fit_transform(df[["rule_category"]])
    else:
        rule_feat = encoder.transform(df[["rule_category"]])
    
    logger.info(f"Rule category features: {rule_feat.shape}")
    
    # Concatenate
    X_full = np.hstack([X_embed, rule_feat])
    logger.info(f"Full feature matrix: {X_full.shape}")
    
    return X_full, encoder


def load_labels(labels_path: Path, df: pd.DataFrame) -> pd.DataFrame:
    """
    Load human-labelled subset and merge with main dataframe.
    
    Expected columns in labels file:
    - shelf_canon: Canonical shelf name
    - human_category: Manual label (one of the 13 categories)
    """
    logger.info(f"Loading human-labelled subset from {labels_path}...")
    lab = pd.read_csv(labels_path)
    
    # Validate required columns
    required_cols = ["shelf_canon", "human_category"]
    missing = [col for col in required_cols if col not in lab.columns]
    if missing:
        raise ValueError(f"Labels file missing required columns: {missing}")
    
    # Merge to attach embeddings & rule features
    df_lab = df.merge(lab, on="shelf_canon", how="inner")
    
    logger.info(f"Found {len(df_lab)} shelves with human labels")
    
    # Log label distribution
    label_dist = df_lab["human_category"].value_counts()
    logger.info("Human label distribution:")
    for cat, count in label_dist.items():
        logger.info(f"  {cat}: {count:,} ({100*count/len(df_lab):.1f}%)")
    
    return df_lab


def train_classifier(X_train: np.ndarray, y_train: np.ndarray, 
                     X_test: np.ndarray, y_test: np.ndarray) -> LogisticRegression:
    """Train logistic regression classifier."""
    logger.info("Training logistic regression classifier...")
    
    clf = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    logger.info("\nClassification report on held-out labelled set:")
    logger.info("\n" + classification_report(y_test, y_pred))
    logger.info("\nConfusion matrix:")
    logger.info("\n" + str(confusion_matrix(y_test, y_pred)))
    
    return clf


def predict_all(df: pd.DataFrame, X_all: np.ndarray, clf: LogisticRegression, 
                threshold: float = 0.7) -> pd.DataFrame:
    """Predict categories for all shelves."""
    logger.info("Predicting categories for ALL shelves...")
    
    y_all_pred = clf.predict(X_all)
    y_all_proba = clf.predict_proba(X_all)
    max_proba = y_all_proba.max(axis=1)
    
    df = df.copy()
    df["pred_category"] = y_all_pred
    df["pred_confidence"] = max_proba
    
    # High-confidence final label, else UNKNOWN_OTHER
    df["final_category"] = np.where(
        df["pred_confidence"] >= threshold,
        df["pred_category"],
        "UNKNOWN_OTHER",
    )
    
    # Log distribution
    pred_dist = df["pred_category"].value_counts()
    logger.info("\nPredicted category distribution:")
    for cat, count in pred_dist.items():
        logger.info(f"  {cat}: {count:,} ({100*count/len(df):.1f}%)")
    
    final_dist = df["final_category"].value_counts()
    logger.info(f"\nFinal category distribution (threshold={threshold}):")
    for cat, count in final_dist.items():
        logger.info(f"  {cat}: {count:,} ({100*count/len(df):.1f}%)")
    
    return df


def inspect_hero_heroine_pairing(df: pd.DataFrame, output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Inspect predictions for HERO, HEROINE, and PAIRING categories.
    
    Exports shelves predicted as these categories for manual review.
    """
    logger.info("\nInspecting HERO/HEROINE/PAIRING predictions...")
    
    focus_categories = ["HERO_ARCHETYPE", "HEROINE_ARCHETYPE", "PAIRING_TYPE"]
    
    # Filter to focus categories
    df_focus = df[df["pred_category"].isin(focus_categories)].copy()
    
    logger.info(f"Found {len(df_focus):,} shelves predicted as HERO/HEROINE/PAIRING")
    
    # Sort by confidence (descending) and count (descending)
    df_focus = df_focus.sort_values(
        by=["pred_confidence", "count"] if "count" in df_focus.columns else ["pred_confidence"],
        ascending=[False, False] if "count" in df_focus.columns else [False]
    )
    
    # Add comparison columns
    df_focus["rule_category"] = df_focus["shelf_canon"].apply(assign_category)
    df_focus["rule_matches_pred"] = df_focus["rule_category"] == df_focus["pred_category"]
    
    # Select columns for export
    export_cols = ["shelf_canon", "pred_category", "pred_confidence", 
                   "rule_category", "rule_matches_pred"]
    if "count" in df_focus.columns:
        export_cols.append("count")
    if "shelf_raw" in df_focus.columns:
        export_cols.append("shelf_raw")
    
    df_export = df_focus[export_cols].copy()
    
    # Log statistics
    logger.info("\nHERO/HEROINE/PAIRING prediction statistics:")
    logger.info(f"  Total predictions: {len(df_focus):,}")
    for cat in focus_categories:
        cat_count = len(df_focus[df_focus["pred_category"] == cat])
        logger.info(f"  {cat}: {cat_count:,}")
    
    rule_match_rate = df_focus["rule_matches_pred"].mean() * 100
    logger.info(f"  Rule matches prediction: {rule_match_rate:.1f}%")
    
    # Export if path provided
    if output_path:
        logger.info(f"\nExporting to {output_path}...")
        df_export.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df_export):,} rows")
    
    return df_export


def save_model(model_path: Path, embedder_name: str, encoder: OneHotEncoder, 
               clf: LogisticRegression):
    """Save model components for later use."""
    logger.info(f"Saving model components to {model_path}...")
    joblib.dump(
        {
            "embedder_name": embedder_name,
            "encoder": encoder,
            "clf": clf,
        },
        model_path,
    )
    logger.info("Model saved successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid shelf classifier combining rule-based and embedding features"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV with shelf_canon column (e.g., shelf_canonical_test.csv)",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Human-labelled CSV with shelf_canon and human_category columns",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV with predictions",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to save model (pickle file)",
    )
    parser.add_argument(
        "--inspect-hero-pairing",
        type=Path,
        default=None,
        help="Optional: Export HERO/HEROINE/PAIRING predictions for review",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Sentence transformer model name",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for final_category (default: 0.7)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size for train/test split (default: 0.2)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for checkpoint files (auto-detected from output path if not specified)",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation even if checkpoints exist",
    )
    
    args = parser.parse_args()
    
    # Ensure output directories exist
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    if args.inspect_hero_pairing:
        args.inspect_hero_pairing.parent.mkdir(parents=True, exist_ok=True)
    
    # Set up checkpoint directory
    if args.checkpoint_dir is None:
        # Auto-detect from output path
        args.checkpoint_dir = args.output.parent / "checkpoints"
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoint directory: {args.checkpoint_dir}")
    
    # Define checkpoint paths
    rule_checkpoint = args.checkpoint_dir / "rule_categories.csv"
    embedding_checkpoint = args.checkpoint_dir / "embeddings.pkl"
    
    # ==========================
    # 1. LOAD DATA & RULE LABEL
    # ==========================
    df = load_data(args.input)
    
    if args.force_recompute:
        logger.info("Force recompute enabled - ignoring checkpoints")
        rule_checkpoint = None
        embedding_checkpoint = None
    
    df = assign_rule_categories(df, checkpoint_path=rule_checkpoint if not args.force_recompute else None)
    
    # ==========================
    # 2. EMBEDDINGS
    # ==========================
    df = compute_embeddings(df, args.embedding_model, 
                           checkpoint_path=embedding_checkpoint if not args.force_recompute else None)
    
    # ==========================
    # 3. ONE-HOT(rule_category)
    # ==========================
    X_full, encoder = prepare_features(df)
    
    # ==========================
    # 4. LOAD HUMAN LABELS
    # ==========================
    df_lab = load_labels(args.labels, df)
    
    # Prepare features for labelled subset
    X_lab_full, _ = prepare_features(df_lab, encoder)
    y_lab = df_lab["human_category"].values
    
    # ==========================
    # 5. TRAIN CLASSIFIER
    # ==========================
    X_train, X_test, y_train, y_test = train_test_split(
        X_lab_full,
        y_lab,
        test_size=args.test_size,
        stratify=y_lab,
        random_state=42,
    )
    
    clf = train_classifier(X_train, y_train, X_test, y_test)
    
    # ==========================
    # 6. PREDICT ALL SHELVES
    # ==========================
    df = predict_all(df, X_full, clf, threshold=args.threshold)
    
    # Save predictions
    logger.info(f"\nSaving predictions to {args.output}...")
    df.to_csv(args.output, index=False)
    logger.info(f"Saved {len(df):,} predictions")
    
    # ==========================
    # 7. INSPECT HERO/HEROINE/PAIRING
    # ==========================
    if args.inspect_hero_pairing:
        inspect_hero_heroine_pairing(df, args.inspect_hero_pairing)
    
    # ==========================
    # 8. SAVE MODEL
    # ==========================
    save_model(args.model, args.embedding_model, encoder, clf)
    
    logger.info("\n" + "="*60)
    logger.info("Classification complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()

