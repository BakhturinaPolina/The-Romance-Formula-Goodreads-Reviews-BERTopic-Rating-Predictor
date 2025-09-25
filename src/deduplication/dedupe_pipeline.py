# File: src/deduplication/dedupe_pipeline.py
from __future__ import annotations

import os, re, sys, time, json, math, psutil, typing as t
from dataclasses import dataclass, asdict, fields, is_dataclass
from pathlib import Path
from collections import defaultdict
import numpy as np
import pyarrow.dataset as ds
import duckdb, pandas as pd
import typer

# ----------------------- Config -----------------------

@dataclass(frozen=True)
class Config:
    in_pairs: Path
    out_dir: Path
    db_path: t.Optional[Path] = None
    base_threshold: float = 0.50
    min_len: int = 3
    max_rank: int | None = None
    block_numeric: bool = True
    # JSON-safe: store patterns as strings; compile on demand
    noise_regexes: tuple[str, ...] = (r"^zz+",)
    hist_bin_start: float = 0.45
    hist_bin_end: float = 1.00
    hist_bin_step: float = 0.01
    sample_edges_per_cluster: int = 5
    batch_size: int = 250_000

    def ensure(self) -> "Config":
        self.out_dir.mkdir(parents=True, exist_ok=True)
        return self

    # Compile lazily for Python-side checks
    def compiled_noise_regexes(self) -> tuple[re.Pattern[str], ...]:
        return tuple(re.compile(pat, re.I) for pat in self.noise_regexes)

# ----------------------- Utils ------------------------

def p(x: str) -> None:
    print(x); sys.stdout.flush()

def head(title: str) -> None:
    sep = "=" * 100
    p(sep); p(f"🔎 {title}"); p(sep)

def human_bytes(n: int) -> str:
    units = ["B","KB","MB","GB","TB"]; i=0; x=float(n)
    while x>=1024 and i<len(units)-1:
        x/=1024; i+=1
    return f"{x:.2f} {units[i]}"

def log_memory() -> None:
    proc = psutil.Process(os.getpid())
    p(f"💾 RSS: {human_bytes(proc.memory_info().rss)}")

def open_duckdb(db_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path) if db_path else ":memory:")

def asdict_json(obj: t.Any) -> t.Any:
    """Dataclass-aware, JSON-safe converter. Paths→str; regex strings kept as-is."""
    if is_dataclass(obj):
        return {f.name: asdict_json(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, (list, tuple)):
        return [asdict_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: asdict_json(v) for k, v in obj.items()}
    return obj  # numbers/bool/str/None

# ----------------------- Dataset / scanning ----------------------

def scan_pairs(cfg: Config) -> ds.Dataset:
    return ds.dataset(str(cfg.in_pairs), format="parquet")

def make_scanner(dsobj: ds.Dataset, columns: list[str], batch_size: int) -> ds.Scanner:
    return dsobj.scanner(columns=columns, batch_size=batch_size)

# ----------------------- Filter rules ---------------------------

def is_short(tok: str, k: int) -> bool:
    return len(tok) < k

def looks_numeric(tok: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:[.,]\d+)?", tok))

def is_noise(tok: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pat.search(tok) for pat in patterns)

def edge_allowed(row: dict[str, t.Any], cfg: Config) -> bool:
    a = str(row["token_a"]); b = str(row["token_b"]); sim = float(row["cosine_sim"])
    if cfg.max_rank is not None and int(row.get("rank", 0)) > cfg.max_rank: return False
    if sim < cfg.base_threshold: return False
    if is_short(a, cfg.min_len) or is_short(b, cfg.min_len): return False
    if cfg.block_numeric and (looks_numeric(a) or looks_numeric(b)): return False
    if is_noise(a, cfg.compiled_noise_regexes()) or is_noise(b, cfg.compiled_noise_regexes()): return False
    return True

# ----------------------- Streaming stats ------------------------

@dataclass
class Stats:
    total_rows: int
    kept_rows: int
    sum_sim: float
    min_sim: float
    max_sim: float
    hist_bins: np.ndarray
    hist_counts: np.ndarray

    @property
    def mean_sim(self) -> float:
        # use kept_rows (post-filter) to avoid wrong denominator
        denom = self.kept_rows if self.kept_rows else 0
        return (self.sum_sim / denom) if denom else float("nan")

def stream_stats(cfg: Config, filtered: bool = False) -> Stats:
    dsobj = scan_pairs(cfg)
    scanner = make_scanner(dsobj, ["token_a","token_b","cosine_sim","rank"], cfg.batch_size)

    nbins = int(math.ceil((cfg.hist_bin_end - cfg.hist_bin_start) / cfg.hist_bin_step))
    bins = np.linspace(cfg.hist_bin_start, cfg.hist_bin_end, nbins+1)
    counts = np.zeros(nbins, dtype=np.int64)

    total = kept = 0
    min_sim, max_sim = 1.0, 0.0
    sum_sim = 0.0

    for batch in scanner.to_batches():
        df = batch.to_pandas()
        sims = df["cosine_sim"].to_numpy()
        if filtered:
            mask = df.apply(lambda r: edge_allowed(r, cfg), axis=1).to_numpy()
            df = df[mask]
            sims = sims[mask]
            kept += int(mask.sum())
        total += len(df) if filtered else len(batch)  # for filtered we count only considered rows
        if len(sims):
            min_sim = min(min_sim, float(np.min(sims)))
            max_sim = max(max_sim, float(np.max(sims)))
            sum_sim += float(np.sum(sims))
            c, _ = np.histogram(sims, bins=bins)
            counts += c

    # For BEFORE stats, kept_rows == total rows; AFTER, kept_rows == kept
    kept_rows = kept if filtered else total
    return Stats(total_rows=total, kept_rows=kept_rows, sum_sim=sum_sim,
                 min_sim=min_sim, max_sim=max_sim, hist_bins=bins, hist_counts=counts)

def print_stats(title: str, s: Stats, cfg: Config) -> None:
    head(title)
    p(json.dumps({
        "rows_total": s.total_rows,
        "rows_considered": s.kept_rows,
        "sim_min": round(s.min_sim, 4),
        "sim_max": round(s.max_sim, 4),
        "sim_mean": round(s.mean_sim, 4),
        "hist": {"bins": s.hist_bins.tolist(), "counts": s.hist_counts.tolist()},
        "cfg": {"threshold": cfg.base_threshold, "min_len": cfg.min_len, "max_rank": cfg.max_rank}
    }, indent=2))
    log_memory()

# ----------------------- Filtering (DuckDB) ----------------------

def _sql_noise_clause(cfg: Config, col: str) -> str:
    if not cfg.noise_regexes:
        return ""
    parts = [f"regexp_matches({col}, '{pat}')" for pat in cfg.noise_regexes]
    return " OR ".join(parts)

def filter_pairs_duckdb(cfg: Config, out_path: Path) -> None:
    con = open_duckdb(cfg.db_path)
    con.execute("INSTALL json; LOAD json;")
    noise_a = _sql_noise_clause(cfg, "token_a")
    noise_b = _sql_noise_clause(cfg, "token_b")
    q = f"""
    COPY (
      SELECT token_a, token_b, cosine_sim, rank
      FROM read_parquet('{cfg.in_pairs.as_posix()}')
      WHERE cosine_sim >= {cfg.base_threshold}
        AND length(token_a) >= {cfg.min_len}
        AND length(token_b) >= {cfg.min_len}
        {"AND rank <= " + str(cfg.max_rank) if cfg.max_rank is not None else ""}
        {"AND NOT (regexp_matches(token_a, '^[0-9]+(?:[.,][0-9]+)?') OR regexp_matches(token_b, '^[0-9]+(?:[.,][0-9]+)?'))" if cfg.block_numeric else ""}
        {"AND NOT (" + noise_a + ")" if noise_a else ""}
        {"AND NOT (" + noise_b + ")" if noise_b else ""}
    ) TO '{out_path.as_posix()}' (FORMAT PARQUET);
    """
    con.execute(q)

# ----------------------- Clustering ------------------------------

class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}
    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x; self.rank[x] = 0; return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.rank[ra] < self.rank[rb]: self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]: self.parent[rb] = ra
        else: self.parent[rb] = ra; self.rank[ra] += 1

def build_clusters(filtered_parquet: Path, sample_per_cluster: int, batch_size: int = 500_000
                   ) -> tuple[dict[str, int], dict[int, list[tuple[str, str, float]]]]:
    dsobj = ds.dataset(str(filtered_parquet), format="parquet")
    scanner = dsobj.scanner(columns=["token_a","token_b","cosine_sim"], batch_size=batch_size)
    uf = UnionFind()
    for batch in scanner.to_batches():
        df = batch.to_pandas()
        for a, b in zip(df["token_a"], df["token_b"]):
            uf.union(str(a), str(b))

    root_to_id: dict[str, int] = {}
    token_cluster: dict[str, int] = {}
    next_id = 1
    for tok in uf.parent.keys():
        r = uf.find(tok)
        if r not in root_to_id:
            root_to_id[r] = next_id; next_id += 1
        token_cluster[tok] = root_to_id[r]

    edge_samples: dict[int, list[tuple[str, str, float]]] = defaultdict(list)
    scanner = dsobj.scanner(columns=["token_a","token_b","cosine_sim"], batch_size=batch_size)
    for batch in scanner.to_batches():
        df = batch.to_pandas()
        for a, b, s in zip(df["token_a"], df["token_b"], df["cosine_sim"]):
            cid = token_cluster.get(str(a))
            lst = edge_samples[cid]
            if len(lst) < sample_per_cluster:  # respect cfg
                lst.append((str(a), str(b), float(s)))
    return token_cluster, edge_samples

def summarize_clusters(token_cluster: dict[str, int],
                       edges: dict[int, list[tuple[str,str,float]]]) -> pd.DataFrame:
    by_cluster: dict[int, list[str]] = defaultdict(list)
    for tok, cid in token_cluster.items():
        by_cluster[cid].append(tok)
    rows: list[dict[str, t.Any]] = []
    for cid, toks in by_cluster.items():
        toks_sorted = sorted(toks, key=len)
        size = len(toks_sorted)
        freq: dict[str, int] = defaultdict(int)
        for a, b, _ in edges.get(cid, []):
            freq[a] += 1; freq[b] += 1
        medoid = max(freq.items(), key=lambda x: x[1])[0] if freq else toks_sorted[0]
        rows.append({
            "cluster_id": cid, "size": size, "medoid": medoid,
            "short_rate": sum(1 for t in toks if len(t)<3)/size,
            "digits_rate": sum(1 for t in toks if looks_numeric(t))/size,
        })
    return pd.DataFrame(rows).sort_values(["size","cluster_id"], ascending=[False, True])

# ----------------------- Artifacts -------------------------------

def save_artifacts(cfg: Config,
                   token_cluster: dict[str, int],
                   edge_samples: dict[int, list[tuple[str,str,float]]]) -> None:
    paths = {
        "filtered_pairs": cfg.out_dir / "pairs_filtered.parquet",
        "clusters_map": cfg.out_dir / "clusters_token_map.parquet",
        "clusters_summary": cfg.out_dir / "clusters_summary.parquet",
        "clusters_edges_sample": cfg.out_dir / "clusters_edges_samples.parquet",
        "meta": cfg.out_dir / "meta.json",
    }

    pd.DataFrame([(tok, cid) for tok, cid in token_cluster.items()],
                 columns=["token","cluster_id"]).to_parquet(paths["clusters_map"], index=False)

    summarize_clusters(token_cluster, edge_samples).to_parquet(paths["clusters_summary"], index=False)

    rows = [{"cluster_id": cid, "token_a": a, "token_b": b, "cosine_sim": s}
            for cid, lst in edge_samples.items() for (a,b,s) in lst]
    pd.DataFrame(rows).to_parquet(paths["clusters_edges_sample"], index=False)

    # JSON-safe meta (Paths→str; regex strings already JSON-safe)
    meta = {
        "config": asdict_json(cfg),
        "stats": {},
        "artifacts": {k: v.as_posix() for k, v in paths.items()},
        "generated_at": time.time(),
    }
    paths["meta"].write_text(json.dumps(meta, indent=2))
    p(f"✅ Saved: {paths['clusters_map'].name}, {paths['clusters_summary'].name}, {paths['clusters_edges_sample'].name}")

# ----------------------- Commands -------------------------------

def cmd_stats(in_pairs: Path, threshold: float = 0.50, min_len: int = 3, max_rank: int | None = None) -> None:
    cfg = Config(in_pairs=in_pairs, out_dir=Path("."), base_threshold=threshold, min_len=min_len, max_rank=max_rank).ensure()
    s0 = stream_stats(cfg, filtered=False); print_stats("BEFORE STATS", s0, cfg)
    s1 = stream_stats(cfg, filtered=True);  print_stats("AFTER (FILTERED) STATS", s1, cfg)

def cmd_filter(in_pairs: Path, out_dir: Path, threshold: float = 0.50, min_len: int = 3, max_rank: int | None = None) -> None:
    cfg = Config(in_pairs=in_pairs, out_dir=out_dir, base_threshold=threshold, min_len=min_len, max_rank=max_rank).ensure()
    head("FILTER PAIRS (DuckDB)")
    filter_pairs_duckdb(cfg, cfg.out_dir / "pairs_filtered.parquet")
    p(f"✅ Wrote {(cfg.out_dir / 'pairs_filtered.parquet')}"); log_memory()

def cmd_cluster(filtered_pairs: Path, out_dir: Path) -> None:
    cfg = Config(in_pairs=filtered_pairs, out_dir=out_dir).ensure()
    head("CLUSTER (Union-Find)")
    token_cluster, sample_edges = build_clusters(filtered_pairs, sample_per_cluster=cfg.sample_edges_per_cluster)
    save_artifacts(cfg, token_cluster, sample_edges)
    log_memory()

def cmd_all(in_pairs: Path, out_dir: Path, threshold: float = 0.50, min_len: int = 3, max_rank: int | None = None) -> None:
    cfg = Config(in_pairs=in_pairs, out_dir=out_dir, base_threshold=threshold, min_len=min_len, max_rank=max_rank).ensure()
    head("BEFORE/AFTER STATS"); cmd_stats(in_pairs, threshold, min_len, max_rank)
    head("FILTER");             cmd_filter(in_pairs, out_dir, threshold, min_len, max_rank)
    head("RE-CLUSTER ON FILTERED")
    token_cluster, sample_edges = build_clusters(cfg.out_dir / "pairs_filtered.parquet",
                                                 sample_per_cluster=cfg.sample_edges_per_cluster)
    save_artifacts(cfg, token_cluster, sample_edges)
    head("DONE")

# ----------------------- CLI ------------------------------------

app = typer.Typer(add_completion=False)

@app.command("stats")
def cli_stats(
    in_pairs: Path = typer.Argument(...),
    threshold: float = typer.Option(0.50),
    min_len: int = typer.Option(3),
    max_rank: int | None = typer.Option(None),
):
    cmd_stats(in_pairs, threshold, min_len, max_rank)

@app.command("filter")
def cli_filter(
    in_pairs: Path = typer.Argument(...),
    out_dir: Path = typer.Argument(...),
    threshold: float = typer.Option(0.50),
    min_len: int = typer.Option(3),
    max_rank: int | None = typer.Option(None),
):
    cmd_filter(in_pairs, out_dir, threshold, min_len, max_rank)

@app.command("cluster")
def cli_cluster(
    filtered_pairs: Path = typer.Argument(...),
    out_dir: Path = typer.Argument(...),
):
    cmd_cluster(filtered_pairs, out_dir)

@app.command("all")
def cli_all(
    in_pairs: Path = typer.Argument(...),
    out_dir: Path = typer.Argument(...),
    threshold: float = typer.Option(0.50),
    min_len: int = typer.Option(3),
    max_rank: int | None = typer.Option(None),
):
    cmd_all(in_pairs, out_dir, threshold, min_len, max_rank)

if __name__ == "__main__":
    app()
