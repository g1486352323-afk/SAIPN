#!/bin/bash
#SBATCH --job-name=meta_explicit
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/data_huawei/gaohaizhen/network/saipn/model/metaverse/logs/%x_%j.out
#SBATCH --error=/data_huawei/gaohaizhen/network/saipn/model/metaverse/logs/%x_%j.err
#SBATCH --chdir=/data_huawei/gaohaizhen/network/saipn/model/compare_plot/pheme

set -euxo pipefail

PY_BIN="/data_huawei/gaohaizhen/.conda/envs/network/bin/python"

META_BASE="/data_huawei/gaohaizhen/network/saipn/model/metaverse"
DATA="${META_BASE}/final_dataset_std_with_tags_filled.csv"
EXPLICIT_OUT="${META_BASE}/output/explicit_metaverse"

mkdir -p "${META_BASE}/logs" "${EXPLICIT_OUT}"

echo "==> [EXPLICIT-META] Building explicit network for metaverse dataset"
"${PY_BIN}" /data_huawei/gaohaizhen/network/saipn/model/compare_plot/pheme/new_explicit.py \
  --resample D \
  --delete-after-hours 720.0 \
  --data-file "${DATA}" \
  --output-dir "${EXPLICIT_OUT}" \
  --prefix explicit \
  --reply-weight 1.0 \
  --quote-weight 1.0 \
  --retweet-weight 0.8 \
  --force \
  --save-snapshots

echo "Done: metaverse explicit network saved to ${EXPLICIT_OUT} (including snapshots/)"
