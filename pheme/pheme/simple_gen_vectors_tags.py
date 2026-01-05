#!/usr/bin/env python3
import pandas as pd
import os
from sentence_transformers import SentenceTransformer

def process_tags_text(text: str) -> str:
    """处理Tags列的文本，确保可以向量化"""
    if not isinstance(text, str) or not text.strip():
        return "unknown"

    # 如果是逗号分隔的标签列表，连接成句子
    if ',' in text:
        tags = [tag.strip() for tag in text.split(',') if tag.strip()]
        # 过滤掉过短的标签
        tags = [tag for tag in tags if len(tag) > 1]
        if tags:
            return ' '.join(tags)
        else:
            return "unknown"
    else:
        # 单个标签
        return text.strip() if len(text.strip()) > 1 else "unknown"

print("开始生成Tags向量 (简化版)")

# 读取数据
input_file = 'charliehebdo_gemini_2_flash_output_fixed_from_cleaned.csv'
output_file = 'embeddin/output_vectors_tags_simple.txt'

print(f"输入文件: {input_file}")
print(f"输出文件: {output_file}")

df = pd.read_csv(input_file, dtype={'raw_value.id_str': str}, keep_default_na=False)
print(f"数据形状: {df.shape}")

# 只处理前1000个样本进行测试
n_samples = min(1000, len(df))
print(f"处理样本数: {n_samples}")

ids = df['raw_value.id_str'].astype(str).str.strip().tolist()[:n_samples]
raw_tags = df['Tags'].fillna("").astype(str).tolist()[:n_samples]

print("处理Tags...")
processed_tags = [process_tags_text(tag) for tag in raw_tags]

print(f"示例处理结果:")
for i in range(min(5, len(processed_tags))):
    print(f"  {i}: '{raw_tags[i]}' -> '{processed_tags[i]}'")

# 加载模型
print("加载SentenceTransformer模型...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("模型加载完成")

# 生成向量
print("开始生成向量...")
batch_size = 32  # 使用小批量

os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, "w", encoding="utf-8") as f:
    for i in range(0, len(processed_tags), batch_size):
        batch_ids = ids[i : i + batch_size]
        batch_txt = processed_tags[i : i + batch_size]

        print(f"处理批次 {i//batch_size + 1}/{(len(processed_tags)-1)//batch_size + 1} ({len(batch_txt)} 样本)")

        vecs = model.encode(batch_txt, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)

        for tid, v in zip(batch_ids, vecs):
            if "." in tid and "e" not in tid and tid.endswith(".0"):
                 tid = tid[:-2]

            vec_str = " ".join(map(str, v.tolist()))
            f.write(f"{tid},{vec_str}\n")

print(f"向量生成完成! 输出文件: {output_file}")
print(f"生成了 {len(processed_tags)} 个向量")


