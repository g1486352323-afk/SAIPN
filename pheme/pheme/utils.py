from pathlib import Path
from typing import Optional, Tuple
import pandas as pd


def read_csv_smart(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    unnamed = [c for c in df.columns if c.lower().startswith("unnamed") or c.strip() == ""]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def detect_columns(df: pd.DataFrame, text_col: Optional[str] = None, id_col: Optional[str] = None) -> Tuple[str, str]:
    if text_col and text_col in df.columns:
        tcol = text_col
    else:
        text_candidates = [
            "raw_value.full_text",
            "full_text",
            "text",
            "content",
            "tweet",
            "message",
            "body",
        ]
        tcol = next((c for c in text_candidates if c in df.columns), None)

    if id_col and id_col in df.columns:
        icol = id_col
    else:
        id_candidates = [
            "raw_value.id_str",
            "id_str",
            "tweet_id",
            "id",
            "status_id",
        ]
        icol = next((c for c in id_candidates if c in df.columns), None)

    if tcol is None or icol is None:
        raise ValueError(f"Cannot find required columns. Available: {list(df.columns)}")

    return tcol, icol


def ensure_str_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str)
