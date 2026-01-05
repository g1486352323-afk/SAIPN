"""
生成无Tags向量 (中文) - 使用原文去除#话题标签
"""
import argparse
import re
from pathlib import Path
import torch
from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer


def remove_hashtags_chinese(text: str) -> str:
    """去除中文和英文的 #话题标签"""
    if not isinstance(text, str):
        return ""
    
    # 去除 #xxx# 格式 (微博话题)
    cleaned = re.sub(r'#[^#]+#', '', text)
    # 去除 #xxx 格式 (Twitter风格)
    cleaned = re.sub(r'#\w+', '', cleaned)
    # 清理多余空格
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned if cleaned else "unknown"


def main():
    parser = argparse.ArgumentParser(description="Generate vectors from raw text (no hashtags)")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-col", default="raw_value.full_text")
    parser.add_argument("--id-col", default="raw_value.id_str")
    parser.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model, device=device)

    print(f"Loading data: {args.input}")
    df = pd.read_csv(args.input, dtype={args.id_col: str})

    ids = df[args.id_col].astype(str).str.strip().tolist()
    raw_texts = df[args.text_col].fillna("").astype(str).tolist()

    # 去除话题标签
    texts = [remove_hashtags_chinese(t) for t in raw_texts]

    print(f"Processing {len(texts)} samples...")
    print("Examples:")
    for i in range(min(3, len(texts))):
        print(f"  Original: {raw_texts[i][:60]}...")
        print(f"  Cleaned:  {texts[i][:60]}...")
        print()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for i in tqdm(range(0, len(texts), args.batch), desc="Embeddings (No Tags)"):
            batch_ids = ids[i:i + args.batch]
            batch_texts = texts[i:i + args.batch]

            vecs = model.encode(batch_texts, batch_size=args.batch, convert_to_numpy=True, show_progress_bar=False)

            for tid, vec in zip(batch_ids, vecs):
                if "." in tid and "e" not in tid and tid.endswith(".0"):
                    tid = tid[:-2]
                vec_str = " ".join(map(str, vec.tolist()))
                f.write(f"{tid},{vec_str}\n")

    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
