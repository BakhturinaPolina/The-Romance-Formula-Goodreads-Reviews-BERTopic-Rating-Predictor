# File: src/deduplication/graph_grow.py
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
import typing as t
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class GraphGrowCfg:
    min_sim: float = 0.75               # safety floor to preserve precision
    min_support: int = 2                # need edges to ≥2 cluster tokens
    require_triangle: bool = True       # at least one triangle with cluster
    attach_k_per_node: int = 2          # keep only top-k attachments per new node
    max_new_nodes_per_cluster: int = 10 # growth guardrail
    iterations: int = 1                 # 1–2 passes recommended

def _normalize_edges(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(columns={"cosine_sim":"cosine_sim"})
    out[["token_a","token_b"]] = out[["token_a","token_b"]].astype("string")
    out["cosine_sim"] = out["cosine_sim"].astype(float)
    return out[["token_a","token_b","cosine_sim"]]

def _build_adjacency(candidates: pd.DataFrame, min_sim: float) -> tuple[dict[str, dict[str,float]], set[tuple[str,str]]]:
    """Adjacency by token with sims, plus undirected edge set for triangle checks (why: O(1) lookups)."""
    cand = _normalize_edges(candidates)
    cand = cand[cand["cosine_sim"] >= min_sim].copy()
    adj: dict[str, dict[str,float]] = defaultdict(dict)
    es: set[tuple[str,str]] = set()
    for a,b,s in cand.itertuples(index=False, name=None):
        if a == b: 
            continue
        # keep best sim per direction
        if (b not in adj[a]) or (s > adj[a][b]):
            adj[a][b] = s
        if (a not in adj[b]) or (s > adj[b][a]):
            adj[b][a] = s
        es.add((a,b)); es.add((b,a))
    return adj, es

def _uf_build(edges: pd.DataFrame) -> dict[str,int]:
    parent: dict[str,str] = {}
    rank: dict[str,int] = {}
    def find(x: str) -> str:
        if x not in parent: parent[x]=x; rank[x]=0; return x
        if parent[x]!=x: parent[x]=find(parent[x])
        return parent[x]
    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra==rb: return
        if rank[ra]<rank[rb]: parent[ra]=rb
        elif rank[ra]>rank[rb]: parent[rb]=ra
        else: parent[rb]=ra; rank[ra]+=1
    for a,b,_ in _normalize_edges(edges).itertuples(index=False, name=None):
        union(a,b)
    root_to_id: dict[str,int] = {}
    tok_to_cid: dict[str,int] = {}
    next_id = 1
    for t in list(parent.keys()):
        r = find(t)
        if r not in root_to_id:
            root_to_id[r]=next_id; next_id+=1
        tok_to_cid[t]=root_to_id[r]
    return tok_to_cid

def _tri_possible(nei: set[str], edge_set: set[tuple[str,str]]) -> bool:
    """Require at least one triangle among the neighbors (why: blocks chainy attachments)."""
    if len(nei) < 2:
        return False
    lst = list(nei)
    for i in range(len(lst)):
        for j in range(i+1, len(lst)):
            u, v = lst[i], lst[j]
            if (u,v) in edge_set or (v,u) in edge_set:
                return True
    return False

def _grow_once(seed_edges: pd.DataFrame,
               candidates: pd.DataFrame,
               cfg: GraphGrowCfg) -> tuple[pd.DataFrame, int]:
    seeds = _normalize_edges(seed_edges)
    adj, e_set = _build_adjacency(candidates, cfg.min_sim)
    # current clusters
    cid = _uf_build(seeds)
    clusters: dict[int, set[str]] = defaultdict(set)
    for t, c in cid.items():
        clusters[c].add(t)

    # candidates neighboring clusters
    new_edges: list[tuple[str,str,float]] = []
    added_nodes_total = 0

    for c, toks in clusters.items():
        toks = set(toks)
        # gather 1-hop neighbors outside the cluster
        out_nei: dict[str, list[tuple[str,float]]] = defaultdict(list)
        for u in toks:
            for v, s in adj.get(u, {}).items():
                if v not in toks:
                    out_nei[v].append((u, s))
        # score and attach new nodes
        added_nodes = 0
        for x, supports in sorted(out_nei.items(), key=lambda kv: (-len(kv[1]), -max(s for _,s in kv[1]))):
            if added_nodes >= cfg.max_new_nodes_per_cluster:
                break
            # support tokens inside cluster
            support_tokens = {u for (u, _) in supports}
            if len(support_tokens) < cfg.min_support:
                continue
            if cfg.require_triangle and not _tri_possible(support_tokens, e_set):
                continue
            # choose strongest attachments to cluster (top-k)
            supports_sorted = sorted(supports, key=lambda us: -us[1])[:cfg.attach_k_per_node]
            for u, s in supports_sorted:
                new_edges.append((x, u, s))
            toks.add(x)  # expand
            added_nodes += 1
            added_nodes_total += 1

    if not new_edges:
        return seeds, 0

    grown = pd.concat(
        [seeds, pd.DataFrame(new_edges, columns=["token_a","token_b","cosine_sim"])],
        ignore_index=True
    ).drop_duplicates(["token_a","token_b"], keep="first")
    return grown, added_nodes_total

def grow_by_triangles(clean_graph: pd.DataFrame,
                      candidates: pd.DataFrame,
                      cfg: GraphGrowCfg) -> pd.DataFrame:
    """Iteratively add high-confidence nodes that create triangles with seed clusters."""
    graph = _normalize_edges(clean_graph)
    for _ in range(cfg.iterations):
        graph2, added = _grow_once(graph, candidates, cfg)
        if added == 0:
            break
        graph = graph2
    return graph
