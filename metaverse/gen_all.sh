#!/bin/bash
#SBATCH --job-name=meta_gen_all
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/data_huawei/gaohaizhen/network/saipn/model/metaverse/logs/%x_%j.out
#SBATCH --error=/data_huawei/gaohaizhen/network/saipn/model/metaverse/logs/%x_%j.err
#SBATCH --chdir=/data_huawei/gaohaizhen/network/saipn/model

set -euxo pipefail
mkdir -p /data_huawei/gaohaizhen/network/saipn/model/metaverse/logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate network
export HF_ENDPOINT=https://hf-mirror.com

# 路径配置
DATA="/data_huawei/gaohaizhen/network/saipn/model/compare_plot/meta/final_dataset_std.csv"
EMB_DIR="/data_huawei/gaohaizhen/network/saipn/model/metaverse/embeddin"
mkdir -p "${EMB_DIR}"

echo "=========================================="
echo "Metaverse: Generate Sentiment + Vectors"
echo "=========================================="

# 1) 生成情感分数 (中文模型)
echo ""
echo "[1/3] Generating sentiment scores (Chinese model)..."
python /data_huawei/gaohaizhen/network/saipn/model/metaverse/gen_sentiment_chinese.py \
    --input "$DATA" \
    --output "${EMB_DIR}/final_with_sentiment.csv" \
    --text-col "raw_value.full_text" \
    --id-col "raw_value.id_str" \
    --batch 64 \
    --device cuda

# 2) 生成 Tags 向量
echo ""
echo "[2/3] Generating Tags vectors..."
python /data_huawei/gaohaizhen/network/saipn/model/ablation/gen_vectors_tags.py \
    --input "$DATA" \
    --output "${EMB_DIR}/output_vectors.txt" \
    --text-col "Tags" \
    --id-col "raw_value.id_str" \
    --model "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" \
    --batch 128 \
    --device cuda

# 3) 生成无Tags向量 (原文去除#hashtag)
echo ""
echo "[3/3] Generating No-Tags vectors (raw text without hashtags)..."
python /data_huawei/gaohaizhen/network/saipn/model/metaverse/gen_vectors_no_tags_chinese.py \
    --input "$DATA" \
    --output "${EMB_DIR}/output_vectors_no_tags.txt" \
    --text-col "raw_value.full_text" \
    --id-col "raw_value.id_str" \
    --model "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" \
    --batch 128 \
    --device cuda

echo ""
echo "=========================================="
echo "Done! Files generated in ${EMB_DIR}:"
echo "  - final_with_sentiment.csv"
echo "  - output_vectors.txt (Tags)"
echo "  - output_vectors_no_tags.txt (Raw text)"
echo "=========================================="
