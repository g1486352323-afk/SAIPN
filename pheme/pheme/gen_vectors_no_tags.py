import argparse
import pandas as pd
import torch
import re
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def remove_hashtags(text):
    """移除文本中的 #Hashtag"""
    if not isinstance(text, str):
        return ""
    # 移除 #号及其后的单词 (例如 #JeSuisCharlie -> 空格)
    return re.sub(r'#\S+', '', text).strip()

def main(args):
    print(f"[Info] 正在读取数据: {args.input}")
    if not os.path.exists(args.input):
        print(f"[Error] 输入文件不存在: {args.input}")
        return

    # 读取 CSV
    df = pd.read_csv(args.input, dtype={'raw_value.id_str': str}, keep_default_na=False)
    
    # 自动寻找文本列
    text_candidates = ["raw_value.full_text", "full_text", "text", "content", "raw_value.text"]
    tcol = next((c for c in text_candidates if c in df.columns), None)
    
    # 自动寻找ID列
    id_candidates = ["raw_value.id_str", "id_str", "id"]
    icol = next((c for c in id_candidates if c in df.columns), None)
    
    if not tcol or not icol:
        print(f"[Error] 找不到文本列或ID列。现有列名: {df.columns.tolist()}")
        return

    print(f"[Info] 使用列 -> 文本: {tcol}, ID: {icol}")
    
    # === 清洗 Tag ===
    print("[Info] 正在清洗 Hashtags...")
    ids = df[icol].astype(str).str.strip().tolist()
    raw_texts = df[tcol].fillna("").astype(str).tolist()
    
    # 这里执行关键操作：去掉 #tag
    clean_texts = [remove_hashtags(t) for t in tqdm(raw_texts, desc="Cleaning Tags")]
    
    # 加载模型
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[Info] 加载 BERT 模型 (all-MiniLM-L6-v2) 到 {device}...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"[Info] 开始生成向量，目标文件: {args.output}")
    batch_size = args.batch_size
    
    with open(args.output, "w", encoding="utf-8") as f:
        for i in tqdm(range(0, len(clean_texts), batch_size), desc="Embedding"):
            batch_ids = ids[i : i + batch_size]
            batch_txt = clean_texts[i : i + batch_size]
            
            # 处理空文本 (防止删完tag后变空串报错)
            batch_txt = [t if len(t.strip()) > 0 else "." for t in batch_txt]
            
            vecs = model.encode(batch_txt, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
            
            for tid, v in zip(batch_ids, vecs):
                vec_str = " ".join(map(str, v.tolist()))
                f.write(f"{tid},{vec_str}\n")
    
    print("[Success] 无 Tag 向量生成完毕。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 这里定义参数，必须和 run.sh 里的写法一致
    parser.add_argument('--input', required=True, help="输入CSV路径")
    parser.add_argument('--output', required=True, help="输出TXT路径")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    main(args)