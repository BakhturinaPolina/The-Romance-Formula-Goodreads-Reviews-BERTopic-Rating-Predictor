# File: src/deduplication/mapping_export.py
from __future__ import annotations

import typing as t
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import typer


app = typer.Typer(add_completion=False, help="Frequency-based medoid export + normalization")


# ------------------------------- Config -------------------------------

@dataclass(frozen=True)
class FreqCfg:
    sep: str = ","
    lowercase: bool = False          # why: only enable if your token_map is lowercased
    strip: bool = True
    drop_empty: bool = True
    chunksize: int = 250_000


# ------------------------------- Utils -------------------------------

def _split_tokens(series: pd.Series, cfg: FreqCfg) -> pd.Series:
    s = series.fillna("")
    parts = s.str.split(cfg.sep)
    # Normalize tokens
    def _norm_list(xs: t.List[str]) -> t.List[str]:
        out = []
        for x in xs:
            y = x.strip() if cfg.strip else x
            if cfg.lowercase:
                y = y.lower()
            if cfg.drop_empty and not y:
                continue
            out.append(y)
        return out
    return parts.apply(lambda xs: _norm_list(xs if isinstance(xs, list) else [xs]))


def _explode_counts(df: pd.DataFrame, columns: t.List[str], cfg: FreqCfg) -> pd.DataFrame:
    # why: stream-friendly counting across multiple columns
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        tokens = _split_tokens(df[col], cfg)
        counts = pd.Series(np.concatenate(tokens.values), dtype="string").value_counts()
        if not counts.empty:
            rows.append(counts.rename("count").rename_axis("token").reset_index())
    if not rows:
        return pd.DataFrame(columns=["token", "count"])
    return (
        pd.concat(rows, ignore_index=True)
        .groupby("token", as_index=False)["count"].sum()
        .astype({"token": "string", "count": "int64"})
    )


def compute_token_frequencies(
    csv_paths: t.List[Path],
    columns: t.List[str],
    cfg: FreqCfg,
    csv_kwargs: t.Dict[str, t.Any] | None = None,
) -> pd.DataFrame:
    """
    Stream CSVs and count tokens found in `columns`.
    Returns: DataFrame[token, count]
    """
    csv_kwargs = csv_kwargs or {}
    agg = None
    for path in csv_paths:
        for chunk in pd.read_csv(path, chunksize=cfg.chunksize, **csv_kwargs):
            cnt = _explode_counts(chunk, columns, cfg)
            if agg is None:
                agg = cnt
            else:
                agg = (
                    pd.concat([agg, cnt], ignore_index=True)
                    .groupby("token", as_index=False)["count"].sum()
                )
    if agg is None:
        agg = pd.DataFrame({"token": pd.Series(dtype="string"), "count": pd.Series(dtype="int64")})
    return agg.sort_values("count", ascending=False, kind="stable").reset_index(drop=True)


def select_frequency_medoids(
    token_map: pd.DataFrame,
    token_freq: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    token_map: columns [token, cluster_id]
    token_freq: columns [token, count]
    Returns:
      - cluster_medoids: [cluster_id, medoid, medoid_freq]
      - token_to_medoid: [token, cluster_id, token_freq, medoid, medoid_freq]
    """
    tm = token_map.astype({"token": "string", "cluster_id": "int64"}).copy()
    tf = token_freq.astype({"token": "string", "count": "int64"}).copy()

    tm = tm.merge(tf, on="token", how="left").rename(columns={"count": "token_freq"})
    tm["token_freq"] = tm["token_freq"].fillna(0).astype("int64")

    # Pick medoid: max frequency; tie-break shortest; then lexicographic
    tm["len"] = tm["token"].str.len()
    medoids = (
        tm.sort_values(["cluster_id", "token_freq", "len", "token"],
                       ascending=[True, False, True, True], kind="stable")
          .drop_duplicates(subset=["cluster_id"], keep="first")
          .rename(columns={"token": "medoid", "token_freq": "medoid_freq"})
          [["cluster_id", "medoid", "medoid_freq"]]
          .reset_index(drop=True)
    )

    mapped = (
        tm.merge(medoids, on="cluster_id", how="left")[["token", "cluster_id", "token_freq", "medoid", "medoid_freq"]]
          .astype({"token": "string", "cluster_id": "int64", "token_freq": "int64"})
          .drop_duplicates()
          .reset_index(drop=True)
    )
    return medoids, mapped


def normalize_columns_with_mapping(
    csv_in: Path,
    csv_out: Path,
    mapping: pd.DataFrame,
    columns: t.List[str],
    cfg: FreqCfg,
    dedupe: bool = True,
    sort_tokens: bool = True,
    csv_kwargs: t.Dict[str, t.Any] | None = None,
) -> None:
    """
    For each column in `columns`, write `<col>_canonical` using token→medoid mapping.
    """
    csv_kwargs = csv_kwargs or {}
    df = pd.read_csv(csv_in, **csv_kwargs)
    m = mapping[["token", "medoid"]].astype({"token": "string", "medoid": "string"}).copy()
    # Optional case handling
    if cfg.lowercase:
        m["token"] = m["token"].str.lower()
        m["medoid"] = m["medoid"].str.lower()

    tok2med = dict(zip(m["token"].tolist(), m["medoid"].tolist()))

    for col in columns:
        if col not in df.columns:
            continue
        tokens = _split_tokens(df[col], cfg)

        canon = []
        for xs in tokens:
            ys = [tok2med.get(x, x) for x in xs]
            if dedupe:
                ys = list(dict.fromkeys(ys))  # stable unique
            if sort_tokens:
                ys = sorted(ys)
            canon.append(cfg.sep.join(ys))

        df[col + "_canonical"] = canon

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)


# ------------------------------- CLI -------------------------------

@app.command("export-medoids")
def cli_export_medoids(
    token_map_parquet: Path = typer.Argument(..., help="Path to clusters_token_map.parquet"),
    out_dir: Path = typer.Argument(..., help="Output directory for mapping artifacts"),
    # frequency source
    csv: t.List[Path] = typer.Option(..., "--csv", help="One or more corpus CSVs to compute token frequencies"),
    columns: t.List[str] = typer.Option(["shelves_str", "genres_str"], "--col", help="Columns to scan for tokens"),
    sep: str = typer.Option(",", help="Token separator"),
    lowercase: bool = typer.Option(False, help="Lowercase tokens before counting/mapping"),
    chunksize: int = typer.Option(250_000, help="CSV chunksize"),
):
    """
    Build frequency-based medoids from corpus columns; export mapping parquet files.
    """
    cfg = FreqCfg(sep=sep, lowercase=lowercase, chunksize=chunksize)
    out_dir.mkdir(parents=True, exist_ok=True)

    token_map = pd.read_parquet(token_map_parquet)
    token_freq = compute_token_frequencies(csv, columns, cfg)
    medoids, mapping = select_frequency_medoids(token_map, token_freq)

    medoids_path = out_dir / "cluster_medoids.parquet"
    mapping_path = out_dir / "token_to_medoid.parquet"
    medoids.to_parquet(medoids_path, index=False)
    mapping.to_parquet(mapping_path, index=False)

    typer.echo(f"✅ Wrote medoids: {medoids_path}")
    typer.echo(f"✅ Wrote mapping: {mapping_path}")


@app.command("normalize-columns")
def cli_normalize_columns(
    csv_in: Path = typer.Argument(..., help="Input CSV to normalize"),
    mapping_parquet: Path = typer.Argument(..., help="token_to_medoid.parquet"),
    csv_out: Path = typer.Argument(..., help="Output normalized CSV"),
    columns: t.List[str] = typer.Option(["shelves_str", "genres_str"], "--col", help="Columns to normalize"),
    sep: str = typer.Option(",", help="Token separator"),
    lowercase: bool = typer.Option(False, help="Lowercase tokens before mapping"),
    dedupe: bool = typer.Option(True, help="Remove duplicates in canonical output"),
    sort_tokens: bool = typer.Option(True, help="Sort canonical tokens"),
):
    """
    Apply token→medoid mapping, producing `<col>_canonical` columns.
    """
    cfg = FreqCfg(sep=sep, lowercase=lowercase)
    mapping = pd.read_parquet(mapping_parquet)
    normalize_columns_with_mapping(csv_in, csv_out, mapping, columns, cfg, dedupe=dedupe, sort_tokens=sort_tokens)
    typer.echo(f"✅ Wrote normalized CSV: {csv_out}")


@app.command("normalize-shelves-genres")
def cli_normalize_shelves_genres(
    token_map_parquet: Path = typer.Argument(..., help="clusters_token_map.parquet"),
    corpus_csv: Path = typer.Argument(..., help="Corpus CSV that has shelves_str and genres_str"),
    out_dir: Path = typer.Argument(..., help="Output directory"),
    sep: str = typer.Option(",", help="Token separator"),
    lowercase: bool = typer.Option(False, help="Lowercase tokens for both steps"),
):
    """
    One-shot convenience: export medoids from `corpus_csv` then normalize it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping_dir = out_dir / "mapping"
    mapping_dir.mkdir(exist_ok=True)

    # 1) Export medoids
    cfg = FreqCfg(sep=sep, lowercase=lowercase)
    token_map = pd.read_parquet(token_map_parquet)
    token_freq = compute_token_frequencies([corpus_csv], ["shelves_str", "genres_str"], cfg)
    medoids, mapping = select_frequency_medoids(token_map, token_freq)
    medoids_path = mapping_dir / "cluster_medoids.parquet"
    mapping_path = mapping_dir / "token_to_medoid.parquet"
    medoids.to_parquet(medoids_path, index=False)
    mapping.to_parquet(mapping_path, index=False)

    # 2) Normalize
    normalized_csv = out_dir / (corpus_csv.stem + ".canonical.csv")
    normalize_columns_with_mapping(corpus_csv, normalized_csv, mapping, ["shelves_str", "genres_str"], cfg)

    typer.echo(f"✅ Medoids: {medoids_path}")
    typer.echo(f"✅ Mapping: {mapping_path}")
    typer.echo(f"✅ Normalized CSV: {normalized_csv}")


if __name__ == "__main__":
    app()
