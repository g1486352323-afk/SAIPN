#!/bin/bash
#SBATCH --job-name=meta_explicit_best
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/data_huawei/gaohaizhen/network/saipn/model/metaverse/logs/%x_%j.out
#SBATCH --error=/data_huawei/gaohaizhen/network/saipn/model/metaverse/logs/%x_%j.err
#SBATCH --chdir=/data_huawei/gaohaizhen/network/saipn/model/compare_plot/meta

set -euxo pipefail

PY_BIN="/data_huawei/gaohaizhen/.conda/envs/network/bin/python"

META_BASE="/data_huawei/gaohaizhen/network/saipn/model/metaverse"
DATA="${META_BASE}/final_dataset_std_with_tags_filled.csv"
EXPLICIT_OUT="${META_BASE}/output/explicit_metaverse_best"

mkdir -p "${META_BASE}/logs" "${EXPLICIT_OUT}"

# 与隐式 best 一致的时间窗口：约 42 天 = 1008 小时
WIN_HOURS=1008.0

echo "==> [EXPLICIT-META-BEST] Building explicit network for metaverse dataset (window=${WIN_HOURS}h)"
"${PY_BIN}" /data_huawei/gaohaizhen/network/saipn/model/compare_plot/meta/new_explicit.py \
  --resample D \
  --delete-after-hours "${WIN_HOURS}" \
  --data-file "${DATA}" \
  --output-dir "${EXPLICIT_OUT}" \
  --prefix explicit \
  --reply-weight 1.0 \
  --quote-weight 0.8 \
  --force \
  --save-daily

echo "Done: metaverse explicit(best window) network saved to ${EXPLICIT_OUT} (including snapshots/)"
