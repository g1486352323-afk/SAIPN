#!/bin/bash
#SBATCH --job-name=meta_gen_tags
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/data_huawei/gaohaizhen/network/saipn/model/metaverse/logs/%x_%j.out
#SBATCH --error=/data_huawei/gaohaizhen/network/saipn/model/metaverse/logs/%x_%j.err
#SBATCH --chdir=/data_huawei/gaohaizhen/network/saipn/model

set -euxo pipefail

PY_BIN="/data_huawei/gaohaizhen/.conda/envs/network/bin/python"

META_BASE="/data_huawei/gaohaizhen/network/saipn/model/metaverse"
DATA="/data_huawei/gaohaizhen/network/saipn/model/metaverse/final_dataset_std_with_tags_filled.csv"
EMB_DIR="${META_BASE}/embeddin"
mkdir -p "${META_BASE}/logs" "${EMB_DIR}"

export HF_ENDPOINT=https://hf-mirror.com

# 1) 生成情感 (基于原文 full_text)
"${PY_BIN}" /data_huawei/gaohaizhen/network/saipn/model/ablation-d2/gen_sentiment.py \
  --input "${DATA}" \
  --output "${EMB_DIR}/final_with_sentiment2.csv" \
  --text-col "raw_value.full_text" \
  --id-col "raw_value.id_str" \
  --model "cardiffnlp/twitter-roberta-base-sentiment-latest" \
  --batch 64 \
  --device cuda

# 2) 生成 Tags 向量 (用于带 Tag 的隐式网络)
"${PY_BIN}" /data_huawei/gaohaizhen/network/saipn/model/ablation-d2/gen_vectors.py \
  --input "${DATA}" \
  --output "${EMB_DIR}/output_vectors_tags.txt" \
  --text-col "Tags" \
  --id-col "raw_value.id_str" \
  --model "sentence-transformers/all-MiniLM-L6-v2" \
  --batch 256 \
  --device cuda

# 3) 生成无 Tag 向量：从正文中移除 #xxx 后再编码 (用于 "no-tags" 消融)
"${PY_BIN}" /data_huawei/gaohaizhen/network/saipn/model/ablation-d2/gen_vectors_no_tags.py \
  --input "${DATA}" \
  --output "${EMB_DIR}/output_vectors_no_tags.txt"

echo "Done: metaverse sentiment + tags vectors + no-tag vectors"
