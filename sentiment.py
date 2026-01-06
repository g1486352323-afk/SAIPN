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
