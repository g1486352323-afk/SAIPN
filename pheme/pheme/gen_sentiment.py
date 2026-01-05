import argparse
import pandas as pd
import torch
import os
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

def _normalize_label(label: str, id2label) -> str:
    s = label.lower()
    if s.startswith("label_"):
        try:
            idx = int(s.split("_")[-1])
            s = str(id2label.get(idx, s)).lower()
        except Exception:
            pass
    return s

def detect_columns_internal(df, text_col=None, id_col=None):
    """内置的简单列名检测逻辑"""
    if text_col and text_col in df.columns:
        tcol = text_col
    else:
        candidates = ["raw_value.text", "text", "raw_value.full_text", "full_text", "content"]
        tcol = next((c for c in candidates if c in df.columns), None)

    if id_col and id_col in df.columns:
        icol = id_col
    else:
        candidates = ["raw_value.id_str", "id_str", "id", "raw_value.id"]
        icol = next((c for c in candidates if c in df.columns), None)
        
    return tcol, icol

def generate_final_with_sentiment(
    input_path: str,
    output_path: str,
    text_col: str = None,
    id_col: str = None,
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    batch_size: int = 32,
    device: str = None,
):
    print(f"[Info] 读取文件: {input_path}")
    if not os.path.exists(input_path):
        print(f"[Error] 输入文件不存在: {input_path}")
        return

    # 读取数据，强制 ID 为字符串
    df = pd.read_csv(input_path, dtype={'raw_value.id_str': str, 'id_str': str}, keep_default_na=False)
    
    tcol, icol = detect_columns_internal(df, text_col, id_col)
    
    if not tcol or not icol:
        print(f"[Error] 无法自动检测到文本列或ID列。当前列: {df.columns.tolist()}")
        return
    print(f"[Info] 使用列 -> 文本: {tcol}, ID: {icol}")

    # 准备输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 设备配置
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_idx = 0 if device == "cuda" else -1
    print(f"[Info] 使用设备: {device} (Index: {device_idx})")

    # 加载模型
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True, device=device_idx)
    except Exception as e:
        print(f"[Error] 模型加载失败: {e}")
        return

    neg, neu, pos = [], [], []
    texts = df[tcol].fillna("").astype(str).tolist()

    # 批处理推理
    for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment Analysis"):
        batch = texts[i : i + batch_size]
        # 防止空文本报错
        batch = [t if len(t.strip()) > 0 else "." for t in batch] 
        
        try:
            outputs = pipe(batch, truncation=True, max_length=512)
            for scores in outputs:
                mapped = {}
                for d in scores:
                    lab = _normalize_label(d["label"], model.config.id2label)
                    mapped[lab] = d["score"]
                neg.append(float(mapped.get("negative", 0.0)))
                neu.append(float(mapped.get("neutral", 0.0)))
                pos.append(float(mapped.get("positive", 0.0)))
        except Exception as e:
            print(f"[Warning] Batch {i} 失败: {e}")
            # 填充默认值防止长度不一致
            for _ in range(len(batch)):
                neg.append(0.0); neu.append(1.0); pos.append(0.0)

    df["negative_probability"] = neg
    df["neutral_probability"] = neu
    df["positive_probability"] = pos

    print(f"[Info] 保存结果到: {output_path}")
    df.to_csv(output_path, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-col", default=None)
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--model", default="cardiffnlp/twitter-roberta-base-sentiment-latest")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], nargs="?")
    args = parser.parse_args()

    generate_final_with_sentiment(
        input_path=args.input,
        output_path=args.output,
        text_col=args.text_col,
        id_col=args.id_col,
        model_name=args.model,
        batch_size=args.batch,
        device=args.device,
    )

if __name__ == "__main__":
    main()