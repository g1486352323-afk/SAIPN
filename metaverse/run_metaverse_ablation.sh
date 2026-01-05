#!/bin/bash
#SBATCH --job-name=meta_ablation
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
VEC_NO_TAGS="${EMB_DIR}/output_vectors_no_tags.txt"
SENTI_FILE="${EMB_DIR}/final_with_sentiment2.csv"

mkdir -p "${META_BASE}/logs" "${OUT_BASE}"

# 最佳参数（来自 metaverse 阈值/窗口/衰减敏感性结果）
TH=0.70           # 相似度阈值
DECAY=336.0       # 时间衰减半衰期（小时）≈ 14 天
WIN=1008.0        # 窗口大小（小时）≈ 42 天
RESAMPLE="D"     # 按天重采样

run_ablation() {
  local name="$1"      # ablation_no_tags / ablation_no_sentiment / ablation_no_time_decay
  local script="$2"    # 对应的 Python 脚本名
  local vec="$3"       # 使用的向量文件

  local meta_out_dir="${OUT_BASE}/${name}_meta"
  mkdir -p "${meta_out_dir}"

  echo "==> [Metaverse Ablation] ${name}: TH=${TH}, decay=${DECAY}h, win=${WIN}h"

  ${PY_BIN} /data_huawei/gaohaizhen/network/saipn/model/ablation-d2/${script} \
    --resample "${RESAMPLE}" \
    --decay-unit-hours "${DECAY}" \
    --delete-after-hours "${WIN}" \
    --score-threshold "${TH}" \
    --data-file "${DATA}" \
    --vector-file "${vec}" \
    --sentiment-file "${SENTI_FILE}"

  # 将 ablation-d2 默认输出拷贝到 metaverse/output 下，便于统一汇总
  local abl_base="/data_huawei/gaohaizhen/network/saipn/model/ablation-d2/outputs"
  local src_dir="${abl_base}/${name}-D"

  if [ -f "${src_dir}/index_gpu.csv" ]; then
    cp "${src_dir}/index_gpu.csv" "${meta_out_dir}/index_gpu.csv"
  fi
  if [ -f "${src_dir}/edge_log.csv" ]; then
    cp "${src_dir}/edge_log.csv" "${meta_out_dir}/edge_log.csv"
  fi

  echo "[Done] ${name} -> ${meta_out_dir}"
}

# 1) 移除 Tags（No Tags）
run_ablation "ablation_no_tags" "ablation_no_tags.py" "${VEC_NO_TAGS}"

# 2) 移除 Sentiment（No Sentiment Dimension，仍用情感做元数据）
run_ablation "ablation_no_sentiment" "ablation_no_sentiment.py" "${VEC_TAGS}"

# 3) 移除时间衰减（No Time Decay）
run_ablation "ablation_no_time_decay" "ablation_no_time_decay.py" "${VEC_TAGS}"

echo "All Metaverse ablation runs finished. Index files are under: ${OUT_BASE}/ablation_no_*_meta/"
