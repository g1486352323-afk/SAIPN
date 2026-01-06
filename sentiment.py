from __future__ import annotations

from typing import Dict, Sequence

import pandas as pd
import numpy as np
from transformers import pipeline


def analyze_sentiment(
    texts: Sequence[str],
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    device: str = "cuda",
) -> Dict[str, float]:
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model_name,
        device=0 if device == "cuda" else -1,
    )
    
    results = {}
    for text in texts:
        try:
            output = sentiment_pipeline(str(text))[0]
            label = output["label"].lower()
            score = output["score"]
            
            if label == "positive":
                results[str(text)] = score
            elif label == "negative":
                results[str(text)] = -score
            else:
                results[str(text)] = 0.0
        except Exception:
            results[str(text)] = 0.0
    
    return results


def build_sentiment_map_from_probabilities(
    df: pd.DataFrame,
    *,
    id_col: str = "id_str",
    pos_col: str = "positive_probability",
    neu_col: str = "neutral_probability",
    neg_col: str = "negative_probability",
) -> Dict[str, float]:
    sentiment_map = {}
    for _, row in df.iterrows():
        try:
            tid = str(row[id_col])
            pos = float(row[pos_col])
            neu = float(row[neu_col])
            neg = float(row[neg_col])
            
            if pos > max(neu, neg):
                val = pos
            elif neg > max(neu, pos):
                val = -neg
            else:
                val = (1 - neu) if pos > neg else -(1 - neu)
            
            sentiment_map[tid] = float(val)
        except Exception:
            continue
    
    return sentiment_map
