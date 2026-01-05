#!/bin/bash
#SBATCH --job-name=meta_pipeline
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=16:00:00
#SBATCH --output=/data_huawei/gaohaizhen/network/saipn/model/metaverse/logs/%x_%j.out
#SBATCH --error=/data_huawei/gaohaizhen/network/saipn/model/metaverse/logs/%x_%j.err
#SBATCH --chdir=/data_huawei/gaohaizhen/network/saipn/model

set -euxo pipefail
mkdir -p /data_huawei/gaohaizhen/network/saipn/model/metaverse/logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate network



set -euo pipefail

BASE_DIR="/data_huawei/gaohaizhen/network/saipn/model/metaverse/output"
# 默认使用 best 显式网络，你可以在运行前改成 explicit_metaverse 等
EXPLICIT_DIR="${BASE_DIR}/explicit_metaverse_best"

# 允许复用外部定义的 PY_BIN，否则回退到系统 python
PY_BIN="${PY_BIN:-python}"

CNLR_PY="/data_huawei/gaohaizhen/network/saipn/model/metaverse/count_CNLR.py"

if [[ ! -f "${CNLR_PY}" ]]; then
  echo "[Error] CNLR script not found: ${CNLR_PY}" >&2
  exit 1
fi

if [[ ! -d "${BASE_DIR}" ]]; then
  echo "[Error] BASE_DIR does not exist: ${BASE_DIR}" >&2
  exit 1
fi

echo "==> BASE_DIR       : ${BASE_DIR}"
echo "==> EXPLICIT_DIR  : ${EXPLICIT_DIR}"
echo "==> CNLR script   : ${CNLR_PY}"
echo

for IMPL_DIR in "${BASE_DIR}"/*; do
  # 只处理目录
  if [[ ! -d "${IMPL_DIR}" ]]; then
    continue
  fi

  name="$(basename "${IMPL_DIR}")"

  # 跳过显式网络目录本身
  if [[ "${name}" == "explicit_metaverse_best" ]] || [[ "${name}" == "explicit_metaverse" ]]; then
    echo "[Skip] ${name} (explicit directory)"
    continue
  fi

  echo "==> [CNLR indegree META] ${name} vs $(basename "${EXPLICIT_DIR}")"

  # metaverse/count_CNLR.py writes outputs under its own script directory (this file's folder),
  # not necessarily the current working directory.
  SCRIPT_DIR="$(cd "$(dirname "${CNLR_PY}")" && pwd)"
  rm -f cnlr_summary.csv cnlr_detailed.csv
  rm -f "${SCRIPT_DIR}/cnlr_summary.csv" "${SCRIPT_DIR}/cnlr_detailed.csv"

  # 运行 CNLR（indegree）
  "${PY_BIN}" "${CNLR_PY}" \
    --explicit-dir "${EXPLICIT_DIR}" \
    --implicit-dir "${IMPL_DIR}" \
    --metric indegree || {
      echo "  [Warn] CNLR failed for ${IMPL_DIR}, skipping move." >&2
      continue
    }

  # 将结果移动到对应的隐式目录下
  if [[ -f "${SCRIPT_DIR}/cnlr_summary.csv" ]]; then
    mv "${SCRIPT_DIR}/cnlr_summary.csv" "${IMPL_DIR}/cnlr_indegree_summary.csv"
  elif [[ -f cnlr_summary.csv ]]; then
    mv cnlr_summary.csv "${IMPL_DIR}/cnlr_indegree_summary.csv"
  fi

  if [[ -f "${SCRIPT_DIR}/cnlr_detailed.csv" ]]; then
    mv "${SCRIPT_DIR}/cnlr_detailed.csv" "${IMPL_DIR}/cnlr_indegree_detailed.csv"
  elif [[ -f cnlr_detailed.csv ]]; then
    mv cnlr_detailed.csv "${IMPL_DIR}/cnlr_indegree_detailed.csv"
  fi

  echo "   Saved: ${IMPL_DIR}/cnlr_indegree_summary.csv"
  echo "          ${IMPL_DIR}/cnlr_indegree_detailed.csv"
  echo

done

echo "[Done] Batch CNLR recompute completed for all implicit dirs under: ${BASE_DIR}"
