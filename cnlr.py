from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def _parse_top_list(cell: Any) -> List[Tuple[str, float]]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    if isinstance(cell, list):
        raw = cell
    else:
        s = str(cell).strip()
        if not s:
            return []
        try:
            import ast
            raw = ast.literal_eval(s)
        except Exception:
            return []

    out: List[Tuple[str, float]] = []
    if not isinstance(raw, (list, tuple)):
        return out

    for item in raw:
        try:
            node_id = str(item[0])
            score = float(item[1])
            out.append((node_id, score))
        except Exception:
            continue
    return out


def parse_first_appearance(filepath: str) -> Dict[str, pd.Timestamp]:
    from pathlib import Path
    
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    df = pd.read_csv(p)
    for col in ["w_value", "top10_percent_nodes"]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV: {filepath}")

    df["w_value"] = pd.to_datetime(df["w_value"], errors="coerce")
    df = df.dropna(subset=["w_value"]).sort_values("w_value").reset_index(drop=True)

    first: Dict[str, pd.Timestamp] = {}
    for _, row in df.iterrows():
        t = row["w_value"]
        top_list = _parse_top_list(row["top10_percent_nodes"])
        for node_id, _ in top_list:
            if node_id not in first:
                first[node_id] = pd.Timestamp(t)
    return first


def calculate_cnlr_from_topk(
    implicit_csv_path: str,
    explicit_csv_path: str,
    *,
    exclude_zero_lag: bool = False,
    min_abs_lag_hours: float = 0.0,
) -> Dict[str, float]:
    dates_imp = parse_first_appearance(implicit_csv_path)
    dates_exp = parse_first_appearance(explicit_csv_path)

    common_nodes = set(dates_imp.keys()) & set(dates_exp.keys())
    if not common_nodes:
        return {}

    cnlr_values = []
    for node in common_nodes:
        t_imp = dates_imp[node]
        t_exp = dates_exp[node]
        lag_h = (t_imp - t_exp) / np.timedelta64(1, "h")
        
        if exclude_zero_lag and lag_h == 0.0:
            continue
        if min_abs_lag_hours > 0 and abs(lag_h) < min_abs_lag_hours:
            continue
            
        duration_hours = (t_exp - min(dates_exp[node] for node in common_nodes)) / np.timedelta64(1, "h")
        if duration_hours <= 0:
            continue
            
        cnlr_u = lag_h / duration_hours
        cnlr_values.append(cnlr_u)

    if not cnlr_values:
        return {}

    return {
        "mean_cnlr": float(np.mean(cnlr_values)),
        "median_cnlr": float(np.median(cnlr_values)),
        "std_cnlr": float(np.std(cnlr_values)),
        "min_cnlr": float(np.min(cnlr_values)),
        "max_cnlr": float(np.max(cnlr_values)),
        "count_nodes": int(len(cnlr_values)),
    }
