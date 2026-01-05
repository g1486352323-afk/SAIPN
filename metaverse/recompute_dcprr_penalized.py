import argparse
import ast
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


def _read_graph_from_edgelist(path: Path) -> nx.DiGraph:
    g = nx.DiGraph()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            s = s.replace("→", " ").replace("-->", " ").replace("->", " ").replace(",", " ")
            toks = s.split()
            if len(toks) < 2:
                continue
            u, v = toks[0], toks[1]
            w = None
            if len(toks) >= 3:
                try:
                    w = float(toks[2])
                except Exception:
                    if toks[2].startswith("{"):
                        try:
                            d = ast.literal_eval(" ".join(toks[2:]))
                            if isinstance(d, dict) and "weight" in d:
                                w = float(d["weight"])
                        except Exception:
                            w = None
            if w is None:
                g.add_edge(u, v)
            else:
                g.add_edge(u, v, weight=w)
    return g


def _detect_communities(
    g: nx.DiGraph,
    resolution: float,
    seed: int,
    min_community_size: int,
    max_communities: Optional[int],
) -> Dict[int, List[str]]:
    if g.number_of_nodes() == 0:
        return {}

    g_und = g.to_undirected()

    comm_sets: List[set]
    try:
        import networkx.algorithms.community as nx_comm

        comm_sets = list(
            nx_comm.louvain_communities(
                g_und,
                weight="weight",
                resolution=resolution,
                seed=seed,
            )
        )
    except Exception:
        try:
            import community as community_louvain

            part = community_louvain.best_partition(g_und, resolution=resolution, random_state=seed)
            tmp: Dict[int, List[str]] = {}
            for node, cid in part.items():
                tmp.setdefault(int(cid), []).append(str(node))
            comm_sets = [set(v) for v in tmp.values()]
        except Exception:
            return {}

    comm_sets = [c for c in comm_sets if len(c) >= min_community_size]
    if max_communities is not None and max_communities > 0:
        comm_sets = sorted(comm_sets, key=len, reverse=True)[:max_communities]

    return {i: list(c) for i, c in enumerate(comm_sets)}


def _compute_dcprr_score(
    g: nx.DiGraph,
    community_nodes: List[str],
    walk_length: int,
    num_walks: int,
    rng: np.random.Generator,
    neighbors_cache: Dict[str, List[str]],
    max_start_nodes: int,
) -> float:
    if len(community_nodes) < 2:
        return 0.0

    start_nodes = community_nodes
    if max_start_nodes > 0 and len(start_nodes) > max_start_nodes:
        idx = rng.choice(len(start_nodes), size=max_start_nodes, replace=False)
        start_nodes = [start_nodes[int(i)] for i in idx]

    comm_set = set(community_nodes)
    retention_rates: List[float] = []

    for node in start_nodes:
        neigh0 = neighbors_cache.get(node)
        if neigh0 is None:
            neigh0 = list(g.neighbors(node))
            neighbors_cache[node] = neigh0
        if not neigh0:
            continue

        total_steps = 0
        retained_steps = 0

        for _ in range(num_walks):
            current = node
            for _ in range(walk_length):
                neigh = neighbors_cache.get(current)
                if neigh is None:
                    neigh = list(g.neighbors(current))
                    neighbors_cache[current] = neigh
                if not neigh:
                    break
                nxt = neigh[int(rng.integers(0, len(neigh)))]
                total_steps += 1
                if nxt in comm_set:
                    retained_steps += 1
                current = nxt

        if total_steps > 0:
            retention_rates.append(retained_steps / total_steps)

    return float(np.mean(retention_rates)) if retention_rates else 0.0


def _compute_penalized_dcprr(
    g: nx.DiGraph,
    communities: Dict[int, List[str]],
    seed: int,
    walk_length: int,
    num_walks: int,
    min_community_size_for_score: int,
    max_start_nodes: int,
    k: float,
    m: float,
    hard_zero_size: int,
) -> float:
    if not communities:
        return 0.0

    rng = np.random.default_rng(seed)
    neighbors_cache: Dict[str, List[str]] = {}
    values: List[float] = []

    for _, nodes in communities.items():
        if len(nodes) < min_community_size_for_score:
            continue
        raw = _compute_dcprr_score(
            g,
            nodes,
            walk_length=walk_length,
            num_walks=num_walks,
            rng=rng,
            neighbors_cache=neighbors_cache,
            max_start_nodes=max_start_nodes,
        )
        size = len(nodes)
        if size <= hard_zero_size:
            values.append(0.0)
        else:
            factor = 1.0 / (1.0 + float(np.exp(-k * (size - m))))
            values.append(float(raw) * factor)

    return float(np.mean(values)) if values else 0.0


def _find_snapshot_file(snapshot_dir: Path, date_str: str) -> Optional[Path]:
    s = str(date_str).strip()
    if not s:
        return None

    dt: Optional[datetime] = None
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        try:
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        except Exception:
            dt = None

    has_explicit_time = (":" in s) or ("T" in s) or (" " in s and len(s.split()) >= 2)

    if dt is not None and has_explicit_time:
        dt_key = dt.strftime("%Y-%m-%d_%H-%M")
        cands = sorted(snapshot_dir.glob(f"implicit_edges_{dt_key}.edgelist"))
        if not cands:
            cands = sorted(snapshot_dir.glob(f"implicit_edges_{dt_key}_*.edgelist"))
        if not cands:
            cands = sorted(snapshot_dir.glob(f"implicit_edges_{dt_key}*.edgelist"))
        if cands:
            return max(cands, key=lambda p: p.name)

    date_key = dt.strftime("%Y-%m-%d") if dt is not None else s.split()[0]
    cands = sorted(snapshot_dir.glob(f"implicit_edges_{date_key}.edgelist"))
    if not cands:
        cands = sorted(snapshot_dir.glob(f"implicit_edges_{date_key}_*.edgelist"))
    if not cands:
        cands = sorted(snapshot_dir.glob(f"implicit_edges_{date_key}*.edgelist"))
    if cands:
        return max(cands, key=lambda p: p.name)

    cands = sorted(snapshot_dir.glob(f"*{s.replace(':', '-').replace(' ', '_')}*.edgelist"))
    if not cands:
        cands = sorted(snapshot_dir.glob(f"*{s.split()[0]}*.edgelist"))
    if not cands:
        return None

    return max(cands, key=lambda p: p.name)


def _recompute_one_run(
    run_dir: Path,
    metric_interval: int,
    walk_length: int,
    num_walks: int,
    seed: int,
    community_resolution: float,
    min_community_size_detect: int,
    min_community_size_score: int,
    max_communities: Optional[int],
    max_start_nodes: int,
    k: float,
    m: float,
    hard_zero_size: int,
    global_dcprr_mode: str,
    skip_if_edges_gt: Optional[float],
    skip_if_nodes_gt: Optional[float],
    inplace: bool,
    out_name: str,
    backup_suffix: str,
) -> Tuple[Path, int]:
    index_path = run_dir / "index_gpu.csv"
    snapshot_dir = run_dir / "snapshots"

    if not index_path.exists():
        raise FileNotFoundError(f"index file not found: {index_path}")
    if not snapshot_dir.exists():
        raise FileNotFoundError(f"snapshot dir not found: {snapshot_dir}")

    with index_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"empty csv: {index_path}")

    header = rows[0]
    if "DCPRR" not in header or "Edges" not in header or "Time" not in header:
        raise RuntimeError(f"unexpected header in {index_path}")

    dcprr_idx = header.index("DCPRR")
    edges_idx = header.index("Edges")
    time_idx = header.index("Time")
    nodes_idx = header.index("Nodes") if "Nodes" in header else None

    out_rows = [header]
    step_rows: List[List[str]] = []
    global_rows_for_dcprr: List[List[str]] = []

    last_dcprr = 0.0
    step_idx = 0
    n_updated = 0

    for r in rows[1:]:
        if not r:
            continue
        if len(r) < len(header):
            r = r + [""] * (len(header) - len(r))

        t = str(r[time_idx])
        if t.startswith("GLOBAL_AVERAGE"):
            out_rows.append(r)
            continue

        step_idx += 1

        try:
            edges = float(r[edges_idx])
        except Exception:
            edges = 0.0

        compute_now = (metric_interval <= 1) or (step_idx == 1) or (step_idx % metric_interval == 0)

        nodes_val: Optional[float] = None
        if nodes_idx is not None:
            try:
                nodes_val = float(r[nodes_idx])
            except Exception:
                nodes_val = None

        if edges <= 0:
            dcprr_val = 0.0
            last_dcprr = dcprr_val
        elif (skip_if_edges_gt is not None and edges > skip_if_edges_gt) or (
            skip_if_nodes_gt is not None and nodes_val is not None and nodes_val > skip_if_nodes_gt
        ):
            dcprr_val = last_dcprr
        elif compute_now:
            snap = _find_snapshot_file(snapshot_dir, t)
            if snap is None:
                dcprr_val = last_dcprr
            else:
                g = _read_graph_from_edgelist(snap)
                comms = _detect_communities(
                    g,
                    resolution=community_resolution,
                    seed=seed,
                    min_community_size=min_community_size_detect,
                    max_communities=max_communities,
                )
                dcprr_val = _compute_penalized_dcprr(
                    g,
                    comms,
                    seed=seed,
                    walk_length=walk_length,
                    num_walks=num_walks,
                    min_community_size_for_score=min_community_size_score,
                    max_start_nodes=max_start_nodes,
                    k=k,
                    m=m,
                    hard_zero_size=hard_zero_size,
                )
                last_dcprr = dcprr_val
                global_rows_for_dcprr.append(r)
        else:
            dcprr_val = last_dcprr

        prev = r[dcprr_idx]
        r[dcprr_idx] = f"{dcprr_val:.4f}"
        if prev != r[dcprr_idx]:
            n_updated += 1

        out_rows.append(r)
        step_rows.append(r)

    global_rows = [i for i, r in enumerate(out_rows) if i > 0 and str(r[time_idx]).startswith("GLOBAL_AVERAGE")]
    if global_rows:
        src_rows = global_rows_for_dcprr if global_rows_for_dcprr else step_rows

        dcprr_vals = np.array([float(r[dcprr_idx]) if r[dcprr_idx] else 0.0 for r in src_rows], dtype=float)

        if global_dcprr_mode == "edge_weighted":
            edges_arr = np.array([float(r[edges_idx]) if r[edges_idx] else 0.0 for r in src_rows], dtype=float)
            total_edges = float(np.sum(edges_arr))
            global_dcprr = float(np.average(dcprr_vals, weights=edges_arr)) if total_edges > 0 else 0.0
        elif global_dcprr_mode == "mean_all":
            global_dcprr = float(np.mean(dcprr_vals)) if dcprr_vals.size > 0 else 0.0
        else:
            valid = dcprr_vals > 1e-6
            global_dcprr = float(np.mean(dcprr_vals[valid])) if np.any(valid) else 0.0

        for gi in global_rows:
            out_rows[gi][dcprr_idx] = f"{global_dcprr:.4f}"

    if inplace:
        backup_path = index_path.with_name(index_path.name + backup_suffix)
        if not backup_path.exists():
            backup_path.write_text(index_path.read_text(encoding="utf-8"), encoding="utf-8")
        out_path = index_path
    else:
        out_path = run_dir / out_name

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(out_rows)

    return out_path, n_updated


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--run-dir", type=str, default=None)
    grp.add_argument("--output-base", type=str, default=None)

    ap.add_argument("--run-glob", type=str, default="*", help="Used with --output-base")

    ap.add_argument("--metric-interval", type=int, default=1)

    ap.add_argument(
        "--global-dcprr-mode",
        type=str,
        default="mean_nonzero",
        choices=["mean_nonzero", "edge_weighted", "mean_all"],
    )

    ap.add_argument("--skip-if-edges-gt", type=float, default=None)
    ap.add_argument("--skip-if-nodes-gt", type=float, default=None)

    ap.add_argument("--walk-length", type=int, default=100)
    ap.add_argument("--num-walks", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--community-resolution", type=float, default=1.0)
    ap.add_argument("--min-community-size-detect", type=int, default=4)
    ap.add_argument("--min-community-size-score", type=int, default=4)
    ap.add_argument("--max-communities", type=int, default=None)

    ap.add_argument("--max-start-nodes", type=int, default=0)

    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--m", type=float, default=6.0)
    ap.add_argument("--hard-zero-size", type=int, default=2)

    ap.add_argument("--inplace", action="store_true")
    ap.add_argument("--out-name", type=str, default="index_gpu_dcprr_penalized.csv")
    ap.add_argument("--backup-suffix", type=str, default=".bak_before_dcprr_penalty")

    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    run_dirs: List[Path] = []
    if args.run_dir:
        run_dirs = [Path(args.run_dir)]
    else:
        base = Path(args.output_base)
        run_dirs = [p for p in sorted(base.glob(args.run_glob)) if p.is_dir()]

    for run_dir in run_dirs:
        try:
            out_path, n_updated = _recompute_one_run(
                run_dir=run_dir,
                metric_interval=int(args.metric_interval),
                walk_length=int(args.walk_length),
                num_walks=int(args.num_walks),
                seed=int(args.seed),
                community_resolution=float(args.community_resolution),
                min_community_size_detect=int(args.min_community_size_detect),
                min_community_size_score=int(args.min_community_size_score),
                max_communities=args.max_communities,
                max_start_nodes=int(args.max_start_nodes),
                k=float(args.k),
                m=float(args.m),
                hard_zero_size=int(args.hard_zero_size),
                global_dcprr_mode=str(args.global_dcprr_mode),
                skip_if_edges_gt=args.skip_if_edges_gt,
                skip_if_nodes_gt=args.skip_if_nodes_gt,
                inplace=bool(args.inplace),
                out_name=str(args.out_name),
                backup_suffix=str(args.backup_suffix),
            )
            print(f"[ok] {run_dir.name}: wrote {out_path} (updated_rows={n_updated})")
        except Exception as e:
            print(f"[err] {run_dir}: {e}")


if __name__ == "__main__":
    main()
