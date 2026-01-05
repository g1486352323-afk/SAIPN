print("[DEBUG] 脚本开始执行")
import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch
from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer
print("[DEBUG] 导入完成")

# ================= 内部集成的工具函数 =================
def detect_columns(df: pd.DataFrame, text_col: Optional[str] = None, id_col: Optional[str] = None) -> Tuple[str, str]:
    """自动检测文本列和ID列"""
    if text_col and text_col in df.columns:
        tcol = text_col
    else:
        # 优先使用Tags列，如果没有则使用full_text
        text_candidates = ["Tags", "raw_value.full_text", "full_text", "text", "content", "tweet", "message", "body"]
        tcol = next((c for c in text_candidates if c in df.columns), None)

    if id_col and id_col in df.columns:
        icol = id_col
    else:
        id_candidates = ["raw_value.id_str", "id_str", "tweet_id", "id", "status_id"]
        icol = next((c for c in id_candidates if c in df.columns), None)

    if tcol is None or icol is None:
        raise ValueError(f"无法找到必要的列。现有列名: {list(df.columns)}")
    return tcol, icol

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

def generate_output_vectors(
    input_path: str,
    output_path: Optional[str] = None,
    text_col: Optional[str] = None,
    id_col: Optional[str] = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 128,
    device: Optional[str] = None,
) -> str:
    print(f"[DEBUG] 开始处理文件: {input_path}")
    print(f"[DEBUG] 参数: text_col={text_col}, id_col={id_col}, model={model_name}, batch_size={batch_size}, device={device}")
    print(f"正在读取文件 (强制字符串模式): {input_path}")

    try:
        df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    except Exception as e:
        print(f"CSV 读取失败: {e}")
        return ""

    unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    tcol, icol = detect_columns(df, text_col=text_col, id_col=id_col)
    print(f"检测到 - 文本列: {tcol}, ID列: {icol}")

    # --- [修改开始] 路径处理逻辑 ---
    if output_path is None:
        # 默认保存到当前脚本所在目录的 embeddin 文件夹下
        output_path = str(Path(__file__).resolve().parent.parent / "embeddin" / "output_vectors_tags.txt")

    out_p = Path(output_path)

    # 如果用户给的是一个已存在的目录，自动追加文件名
    if out_p.is_dir():
        out_p = out_p / "output_vectors_tags.txt"
        print(f"[提示] 输出路径是一个目录，已自动调整为: {out_p}")

    out_p.parent.mkdir(parents=True, exist_ok=True)
    # --- [修改结束] ---

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"加载模型: {model_name} 到 {device}")
    model = SentenceTransformer(model_name, device=device)

    ids = df[icol].astype(str).str.strip().tolist()
    raw_texts = df[tcol].fillna("").astype(str).tolist()

    # 处理Tags文本
    texts = []
    for text in raw_texts:
        processed = process_tags_text(text)
        texts.append(processed)

    print(f"开始生成向量，共 {len(texts)} 条...")
    print(f"示例处理结果:")
    for i in range(min(3, len(texts))):
        print(f"  原始: {raw_texts[i][:50]}...")
        print(f"  处理: {texts[i]}")

    with open(out_p, "w", encoding="utf-8") as f:
        for i in tqdm(range(0, len(texts), batch_size), desc="Embeddings"):
            batch_ids = ids[i : i + batch_size]
            batch_txt = texts[i : i + batch_size]

            vecs = model.encode(batch_txt, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)

            for tid, v in zip(batch_ids, vecs):
                if "." in tid and "e" not in tid and tid.endswith(".0"):
                     tid = tid[:-2]

                vec_str = " ".join(map(str, v.tolist()))
                f.write(f"{tid},{vec_str}\n")

    return str(out_p)


def main():
    ap = argparse.ArgumentParser(description="基于Tags列生成向量 (专门用于语义标签网络构建)")
    ap.add_argument("--input", required=True, help="输入CSV文件路径")
    ap.add_argument("--output", default=None, help="输出TXT文件路径 (默认: embeddin/output_vectors_tags.txt)")
    ap.add_argument("--text-col", default=None, help="文本列名 (默认自动检测，优先Tags列)")
    ap.add_argument("--id-col", default=None, help="ID列名 (默认自动检测)")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer模型名")
    ap.add_argument("--batch", type=int, default=128, help="批处理大小")
    ap.add_argument("--device", default=None, choices=[None, "cpu", "cuda"], nargs="?", help="计算设备")
    args = ap.parse_args()

    out = generate_output_vectors(
        input_path=args.input,
        output_path=args.output,
        text_col=args.text_col,
        id_col=args.id_col,
        model_name=args.model,
        batch_size=args.batch,
        device=args.device,
    )
    print(f"Done! Output saved at: {out}")


if __name__ == "__main__":
    main()
