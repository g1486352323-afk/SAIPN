"""
中文情感分析 - 用于 Metaverse 数据集
"""
import argparse
from pathlib import Path
import torch
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

def main():
    parser = argparse.ArgumentParser(description="Chinese Sentiment Analysis")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-col", default="raw_value.full_text")
    parser.add_argument("--id-col", default="raw_value.id_str")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载中文情感模型
    model_name = "uer/roberta-base-finetuned-chinanews-chinese"
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    # 读取数据
    print(f"Loading data: {args.input}")
    df = pd.read_csv(args.input, dtype={args.id_col: str})
    
    texts = df[args.text_col].fillna("").astype(str).tolist()
    ids = df[args.id_col].astype(str).tolist()

    print(f"Processing {len(texts)} samples...")

    results = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), args.batch), desc="Sentiment"):
            batch_texts = texts[i:i + args.batch]
            batch_ids = ids[i:i + args.batch]

            # Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)

            # Forward
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()

            # 模型输出: [negative, positive] 或 [negative, neutral, positive]
            # 根据模型调整
            for j, tid in enumerate(batch_ids):
                if probs.shape[1] == 2:
                    # 二分类: [neg, pos]
                    neg_prob = float(probs[j, 0])
                    pos_prob = float(probs[j, 1])
                    neu_prob = 0.0
                elif probs.shape[1] == 3:
                    # 三分类: [neg, neu, pos]
                    neg_prob = float(probs[j, 0])
                    neu_prob = float(probs[j, 1])
                    pos_prob = float(probs[j, 2])
                else:
                    neg_prob, neu_prob, pos_prob = 0.33, 0.34, 0.33

                results.append({
                    args.id_col: tid,
                    'positive_probability': pos_prob,
                    'neutral_probability': neu_prob,
                    'negative_probability': neg_prob
                })

    # 保存结果
    result_df = pd.DataFrame(results)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output, index=False)
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
