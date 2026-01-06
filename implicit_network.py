from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch

from .dcprr import calculate_all_dcprr_scores
from .metrics import compute_basic_graph_metrics, detect_communities_louvain


@dataclass(frozen=True)
class ImplicitNetworkConfig:
    resample: str = "H"
    decay_unit_hours: float = 48.0
    delete_after_hours: float = 72.0
    score_threshold: float = 0.70
    metric_interval: int = 1
    max_nodes: int = 80000
    add_sentiment_dim: bool = True
    compute_dcprr: bool = True


def load_vectors_to_device(
    file_path: str,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float16,
) -> Dict[str, torch.Tensor]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vectors: Dict[str, torch.Tensor] = {}

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(",", 1)
            if len(parts) < 2:
                continue
            key = parts[0].strip()
            raw = parts[1].strip()
            if not key or not raw:
                continue
            try:
                vec = torch.tensor([float(x) for x in raw.split()], dtype=dtype, device=device)
            except Exception:
                continue
            vectors[key] = vec

    return vectors


def build_sentiment_map_from_probabilities(
    df: pd.DataFrame,
    *,
    id_col: str = "id_str",
    pos_col: str = "positive_probability",
    neu_col: str = "neutral_probability",
    neg_col: str = "negative_probability",
) -> Dict[str, float]:
    m: Dict[str, float] = {}
    for _, row in df.iterrows():
        try:
            tid = str(row[id_col])
        except Exception:
            continue

        try:
            pos = float(row[pos_col])
            neu = float(row[neu_col])
            neg = float(row[neg_col])
        except Exception:
            continue

        if pos > max(neu, neg):
            val = pos
        elif neg > max(neu, pos):
            val = -neg
        else:
            val = (1 - neu) if pos > neg else -(1 - neu)

        m[tid] = float(val)

    return m


def _normalize(v: torch.Tensor) -> torch.Tensor:
    n = torch.norm(v)
    if n.item() == 0:
        return v
    return v / n


def build_implicit_network(
    events: pd.DataFrame,
    vector_dict: Mapping[str, torch.Tensor],
    *,
    sentiment_map: Optional[Mapping[str, float]] = None,
    id_col: str = "raw_value.id_str",
    time_col: str = "raw_value.created_at",
    config: Optional[ImplicitNetworkConfig] = None,
    device: Optional[torch.device] = None,
) -> Tuple[nx.DiGraph, List[Dict[str, Any]]]:
    cfg = config or ImplicitNetworkConfig()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sentiment_map = sentiment_map or {}

    df = events.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df.sort_values(time_col, inplace=True)
    df.set_index(time_col, inplace=True, drop=False)

    G = nx.DiGraph()

    history_ids: List[str] = []
    history_count = 0

    sample_any = next(iter(vector_dict.values()), None)
    base_dim = int(sample_any.shape[0]) if sample_any is not None else 384
    dim = base_dim + (1 if cfg.add_sentiment_dim else 0)

    history_tensor = torch.zeros((int(cfg.max_nodes), dim), dtype=torch.float16, device=device)
    history_times = torch.zeros(int(cfg.max_nodes), dtype=torch.float32, device=device)
    history_emotions = torch.zeros(int(cfg.max_nodes), dtype=torch.float16, device=device)

    prev_edges_snapshot: Set[Tuple[str, str]] = set()
    metrics_rows: List[Dict[str, Any]] = []

    groups = list(df.resample(cfg.resample))

    for step_idx, (w, w_t) in enumerate(groups, start=1):
        if len(w_t) == 0:
            continue

        current_ts = float(w.timestamp())
        cutoff_ts = current_ts - float(cfg.delete_after_hours) * 3600.0

        if G.number_of_nodes() > 0:
            nodes_to_remove: List[str] = []
            for node, create_time in G.nodes(data="time"):
                if create_time is None:
                    continue
                if float(create_time) < cutoff_ts:
                    nodes_to_remove.append(str(node))
            if nodes_to_remove:
                G.remove_nodes_from(nodes_to_remove)

        batch_ids: List[str] = []
        batch_vecs: List[torch.Tensor] = []
        batch_times: List[float] = []
        batch_emos: List[float] = []

        for _, row in w_t.iterrows():
            tid = str(row[id_col])
            ts = float(pd.to_datetime(row[time_col]).timestamp())
            emo = float(sentiment_map.get(tid, 0.0))

            vec0 = vector_dict.get(tid)
            if vec0 is None:
                continue

            vec = vec0.to(device=device, dtype=torch.float16)
            if cfg.add_sentiment_dim:
                e = torch.tensor([emo], device=device, dtype=torch.float16)
                vec = torch.cat((vec, e))

            vec = _normalize(vec)

            batch_ids.append(tid)
            batch_vecs.append(vec)
            batch_times.append(ts)
            batch_emos.append(emo)

            G.add_node(tid, sentiment=float(emo), time=ts)

        if not batch_ids:
            continue

        batch_tensor = torch.stack(batch_vecs)
        batch_time_tensor = torch.tensor(batch_times, device=device, dtype=torch.float32)

        weights_hist = None
        h_mask = None
        if history_count > 0:
            curr_hist_tensor = history_tensor[:history_count]
            curr_hist_times = history_times[:history_count]

            sim_h = torch.mm(batch_tensor, curr_hist_tensor.t())
            diff_h = (batch_time_tensor.unsqueeze(1) - curr_hist_times.unsqueeze(0)) / 3600.0
            decay_h = torch.exp(-torch.abs(diff_h) / max(float(cfg.decay_unit_hours), 0.01))
            valid_time_mask = (curr_hist_times >= cutoff_ts).unsqueeze(0)
            scores_h = sim_h * decay_h * valid_time_mask

            h_mask = scores_h > float(cfg.score_threshold)
            weights_hist = scores_h

        sim_in = torch.mm(batch_tensor, batch_tensor.t())
        tril_mask = torch.tril(torch.ones_like(sim_in), diagonal=-1).bool()
        diff_in = (batch_time_tensor.unsqueeze(1) - batch_time_tensor.unsqueeze(0)) / 3600.0
        decay_in = torch.exp(-torch.abs(diff_in) / max(float(cfg.decay_unit_hours), 0.01))
        scores_in = sim_in * decay_in * tril_mask
        in_mask = scores_in > float(cfg.score_threshold)

        batch_final_emos: List[float] = []
        batch_final_vecs: List[torch.Tensor] = []

        for i, tid in enumerate(batch_ids):
            current_emo = float(batch_emos[i])
            vol_update = 0.0

            if h_mask is not None and weights_hist is not None:
                valid_h = h_mask[i]
                if bool(valid_h.any()):
                    w_h = weights_hist[i][valid_h]
                    e_h = history_emotions[:history_count][valid_h]
                    vol_update += torch.mean(w_h * e_h).item()

                    h_idx_list = torch.nonzero(valid_h).squeeze(1).cpu().tolist()
                    w_list = w_h.detach().cpu().tolist()
                    for k, h_idx in enumerate(h_idx_list):
                        src_id = history_ids[int(h_idx)]
                        if G.has_node(src_id):
                            G.add_edge(src_id, tid, weight=float(w_list[k]))

            valid_in = in_mask[i]
            if bool(valid_in.any()):
                w_in = scores_in[i][valid_in]
                src_indices = torch.nonzero(valid_in).squeeze(1)
                src_vals = [batch_final_emos[int(idx)] for idx in src_indices.cpu().tolist()]
                src_tensor = torch.tensor(src_vals, device=device, dtype=torch.float16)
                vol_update += torch.mean(w_in * src_tensor).item()

                src_idx_list = src_indices.cpu().tolist()
                w_list_in = w_in.detach().cpu().tolist()
                for k, src_idx in enumerate(src_idx_list):
                    src_id = batch_ids[int(src_idx)]
                    G.add_edge(src_id, tid, weight=float(w_list_in[k]))

            new_emo = max(-1.0, min(1.0, current_emo + float(vol_update)))
            batch_final_emos.append(float(new_emo))

            if cfg.add_sentiment_dim:
                bert_part = batch_tensor[i][:-1]
                emo_t = torch.tensor([new_emo], device=device, dtype=torch.float16)
                new_vec = torch.cat((bert_part, emo_t))
            else:
                new_vec = batch_tensor[i]

            new_vec = _normalize(new_vec)
            batch_final_vecs.append(new_vec)
            G.nodes[tid]["sentiment"] = float(new_emo)

        n_new = len(batch_ids)
        if history_count + n_new < int(cfg.max_nodes):
            start = history_count
            end = history_count + n_new
            history_tensor[start:end] = torch.stack(batch_final_vecs)
            history_times[start:end] = batch_time_tensor
            history_emotions[start:end] = torch.tensor(batch_final_emos, device=device, dtype=torch.float16)
            history_ids.extend(batch_ids)
            history_count += n_new

        compute_metrics_now = (step_idx == 1) or (step_idx % int(cfg.metric_interval) == 0)
        if compute_metrics_now:
            comms = detect_communities_louvain(G)
            m, prev_edges_snapshot = compute_basic_graph_metrics(
                G,
                communities=comms,
                prev_edges_snapshot=prev_edges_snapshot,
                compute_cnlr=True,
            )

            if cfg.compute_dcprr and comms:
                try:
                    dc_scores = calculate_all_dcprr_scores(G, comms)
                    m["dcprr"] = float(np.mean(list(dc_scores.values()))) if dc_scores else 0.0
                except Exception:
                    m["dcprr"] = 0.0
            else:
                m["dcprr"] = 0.0

            m["time"] = pd.Timestamp(w).to_pydatetime()
            metrics_rows.append(m)

    return G, metrics_rows
