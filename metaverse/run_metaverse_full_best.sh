#!/bin/bash
#SBATCH --job-name=meta_full_best
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=/data_huawei/gaohaizhen/network/saipn/model/metaverse/logs/%x_%j.out
#SBATCH --error=/data_huawei/gaohaizhen/network/saipn/model/metaverse/logs/%x_%j.err
#SBATCH --chdir=/data_huawei/gaohaizhen/network/saipn/model

set -euxo pipefail

PY_BIN="/data_huawei/gaohaizhen/.conda/envs/network/bin/python"

META_BASE="/data_huawei/gaohaizhen/network/saipn/model/metaverse"
OUT_BASE="${META_BASE}/output"
EMB_DIR="${META_BASE}/embeddin"
DATA="${META_BASE}/final_dataset_std_with_tags_filled.csv"
VEC_TAGS="${EMB_DIR}/output_vectors_tags.txt"
SENTI_FILE="${EMB_DIR}/final_with_sentiment2.csv"
EXPLICIT_DIR="${OUT_BASE}/explicit_metaverse_best"

mkdir -p "${META_BASE}/logs" "${OUT_BASE}"

# 最佳参数（与阈值/窗口/衰减敏感性分析中选择的一致）
TH=0.70        # 相似度阈值
DECAY=336.0    # 时间衰减半衰期（小时）≈ 14 天
WIN=1008.0     # 删除窗口大小（小时）≈ 42 天
RESAMPLE="D"  # 按天重采样

OUT_DIR="${OUT_BASE}/full_best_meta"
mkdir -p "${OUT_DIR}"

echo "==> [Metaverse FULL] best params: th=${TH}, decay=${DECAY}h, win=${WIN}h"

"${PY_BIN}" /data_huawei/gaohaizhen/network/saipn/model/ablation-d2/full.py \
  --resample "${RESAMPLE}" \
  --decay-unit-hours "${DECAY}" \
  --delete-after-hours "${WIN}" \
  --score-threshold "${TH}" \
  --data-file "${DATA}" \
  --vector-file "${VEC_TAGS}" \
  --sentiment-file "${SENTI_FILE}" \
  --output-dir "${OUT_DIR}" \
  --metric-interval 7

echo "==> [CNLR indegree META FULL] explicit_metaverse vs full_best_meta"
"${PY_BIN}" /data_huawei/gaohaizhen/network/saipn/model/compare_plot/pheme/count_CNLR.py \
  --explicit-dir "${EXPLICIT_DIR}" \
  --implicit-dir "${OUT_DIR}" \
  --metric indegree || true

[ -f cnlr_summary.csv ] && mv cnlr_summary.csv "${OUT_DIR}/cnlr_indegree_summary.csv"
[ -f cnlr_detailed.csv ] && mv cnlr_detailed.csv "${OUT_DIR}/cnlr_indegree_detailed.csv"

echo "[Done] Metaverse FULL(best) run finished. Output under: ${OUT_DIR}"
