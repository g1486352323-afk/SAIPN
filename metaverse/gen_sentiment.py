import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

# 依然导入 detect_columns 用于自动寻找列名，但不再使用 read_csv_smart
from utils import detect_columns

def _normalize_label(label: str, id2label) -> str:
    s = label.lower()
    if s.startswith("label_"):
        try:
            idx = int(s.split("_")[-1])
            s = str(id2label.get(idx, s)).lower()
        except Exception:
            pass
    return s

def generate_final_with_sentiment(
    input_path: str,
    output_path: Optional[str] = None,
    text_col: Optional[str] = None,
    id_col: Optional[str] = None,
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    batch_size: int = 32,
    device: Optional[str] = None,
) -> str:
    print(f"正在读取文件 (强制字符串模式): {input_path}")
    # [关键修改] dtype=str 防止 ID 变成科学计数法
    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    
    # 自动检测列名
    tcol, icol = detect_columns(df, text_col=text_col, id_col=id_col)
    print(f"检测到 - 文本列: {tcol}, ID列: {icol}")

    if output_path is None:
        output_path = str(Path(__file__).resolve().parent / "embedding" / "final_with_sentiment2.csv")
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_idx = 0 if device == "cuda" else -1
    print(f"使用设备: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True, device=device_idx)

    neg, neu, pos = [], [], []
    texts = df[tcol].fillna("").astype(str).tolist()

    for i in tqdm(range(0, len(texts), batch_size), desc="sentiment"):
        batch = texts[i : i + batch_size]
        # 处理空文本防止报错
        batch = [t if len(t.strip()) > 0 else "." for t in batch] 
        
        outputs = pipe(batch, truncation=True, max_length=256)
        for scores in outputs:
            mapped = {}
            for d in scores:
                lab = _normalize_label(d["label"], model.config.id2label)
                mapped[lab] = d["score"]
            neg.append(float(mapped.get("negative", 0.0)))
            neu.append(float(mapped.get("neutral", 0.0)))
            pos.append(float(mapped.get("positive", 0.0)))

    df["negative_probability"] = neg
    df["neutral_probability"] = neu
    df["positive_probability"] = pos

    print(f"正在保存情感结果到: {out_p}")
    df.to_csv(out_p, index=False)
    return str(out_p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--text-col", default=None)
    ap.add_argument("--id-col", default=None)
    ap.add_argument("--model", default="cardiffnlp/twitter-roberta-base-sentiment-latest")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"], nargs="?")
    args = ap.parse_args()

    out = generate_final_with_sentiment(
        input_path=args.input,
        output_path=args.output,
        text_col=args.text_col,
        id_col=args.id_col,
        model_name=args.model,
        batch_size=args.batch,
        device=args.device,
    )
    print(out)

if __name__ == "__main__":
    main()