import argparse
from pathlib import Path
import torch
from tqdm import tqdm
import pandas as pd
import re
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# ================= 配置 =================
# 显式定义 Base Dir，防止相对路径在不同执行目录下出错
BASE_DIR = Path('/data_huawei/gaohaizhen/network/saipn/model/ablation-d2')

# 输入文件默认路径（可被命令行参数覆盖）
INPUT_CSV = Path('/data_huawei/gaohaizhen/network/saipn/model/ablation-d2/final_with_sentiment.csv')

# 输出文件默认路径 (保存到 embedding 目录，可被命令行参数覆盖)
OUTPUT_VEC = BASE_DIR / 'embedding' / 'output_vectors_no_tags.txt'
# =======================================

def remove_hashtags(text):
    """移除文本中的 #Hashtag"""
    if not isinstance(text, str):
        return ""
    # 移除 #号及其后的单词 (例如 #JeSuisCharlie -> 空格)
    # 也就是只保留纯文本内容，测试 Tag 对结构的影响
    return re.sub(r'#\S+', '', text).strip()

def generate_no_tag_vectors(input_csv: Path = INPUT_CSV, output_vec: Path = OUTPUT_VEC):
    print(f"Input CSV:  {input_csv}")
    print(f"Output Vec: {output_vec}")
    
    if not input_csv.exists():
        print(f"[Error] 输入文件不存在: {input_csv}")
        return

    print(f"正在读取数据...")
    df = pd.read_csv(input_csv, dtype={'raw_value.id_str': str}, keep_default_na=False)
    
    # 自动寻找列名
    tcol = next((c for c in ["raw_value.full_text", "full_text", "text", "content", "raw_value.text"] if c in df.columns), None)
    icol = next((c for c in ["raw_value.id_str", "id_str", "id"] if c in df.columns), None)
    
    if not tcol or not icol:
        print(f"[Error] 找不到必要的列。当前列: {df.columns.tolist()}")
        return

    print(f"检测到 - 文本列: {tcol}, ID列: {icol}")
    
    # === [核心步骤] 清洗 Tag ===
    print("正在清洗 Hashtags...")
    ids = df[icol].astype(str).str.strip().tolist()
    raw_texts = df[tcol].fillna("").astype(str).tolist()
    
    # 移除 Tag
    clean_texts = []
    for t in tqdm(raw_texts, desc="Cleaning Tags"):
        cleaned = remove_hashtags(t)
        # 如果清洗后为空 (比如推文全是Tag)，给个占位符防止报错
        if not cleaned: 
            cleaned = "." 
        clean_texts.append(cleaned)
    
    # 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"加载 SentenceTransformer 模型到 {device}...")
    # 使用通用的轻量级模型，或者换成你之前用的 bert-base-multilingual-cased
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    
    # 确保输出目录存在
    output_vec.parent.mkdir(parents=True, exist_ok=True)
    
    # 批量生成与写入
    batch_size = 256
    total = len(clean_texts)
    
    print(f"开始生成无 Tag 向量 (Batch Size: {batch_size})...")
    
    with open(output_vec, "w", encoding="utf-8") as f:
        for i in tqdm(range(0, total, batch_size), desc="Embedding"):
            batch_ids = ids[i : i + batch_size]
            batch_txt = clean_texts[i : i + batch_size]
            
            # 生成向量 (numpy array)
            vecs = model.encode(batch_txt, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
            
            # 写入文件
            for tid, v in zip(batch_ids, vecs):
                # 格式: id,val1 val2 val3 ...
                vec_str = " ".join([f"{x:.6f}" for x in v])
                f.write(f"{tid},{vec_str}\n")
    
    print(f"✅ Done! 无 Tag 向量已保存至: {output_vec}")
    print("👉 现在你可以运行 ablation_no_tags.py 了")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(INPUT_CSV))
    parser.add_argument("--output", default=str(OUTPUT_VEC))
    args = parser.parse_args()

    generate_no_tag_vectors(input_csv=Path(args.input), output_vec=Path(args.output))