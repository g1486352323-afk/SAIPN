#!/usr/bin/env python3
import pandas as pd
import os

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

print("开始调试Tags处理")

# 读取数据
input_file = 'charliehebdo_gemini_2_flash_output_fixed_from_cleaned.csv'
print(f"读取文件: {input_file}")

df = pd.read_csv(input_file, dtype={'raw_value.id_str': str}, keep_default_na=False)
print(f"数据形状: {df.shape}")

# 处理Tags
raw_tags = df['Tags'].fillna("").astype(str).tolist()[:10]  # 只处理前10个
processed_tags = []

print("处理Tags:")
for i, tag in enumerate(raw_tags):
    processed = process_tags_text(tag)
    processed_tags.append(processed)
    print(f"  {i}: '{tag}' -> '{processed}'")

print("Tags处理完成")

print("调试完成 - Tags处理成功")
