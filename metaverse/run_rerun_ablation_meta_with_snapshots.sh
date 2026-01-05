#!/bin/bash
#SBATCH --job-name=meta_ablation_rerun
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

mkdir -p /data_huawei/gaohaizhen/network/saipn/model/metaverse/logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate network

PY_BIN="${PY_BIN:-/data_huawei/gaohaizhen/.conda/envs/network/bin/python}"

META_BASE="/data_huawei/gaohaizhen/network/saipn/model/metaverse"
OUT_BASE="${META_BASE}/output"

DATA="${META_BASE}/final_dataset_std_with_tags_filled.csv"
VEC_TAGS="${META_BASE}/embeddin/output_vectors_tags.txt"
VEC_NO_TAGS="${META_BASE}/embeddin/output_vectors_no_tags.txt"
SENTI_FILE="${META_BASE}/embeddin/final_with_sentiment2.csv"
SENTI_NEUTRAL_FILE="${META_BASE}/embeddin/final_with_sentiment_neutral.csv"

# Explicit (for CNLR + compare plots)
EXPLICIT_DIR="${OUT_BASE}/explicit_metaverse_best"

FULL_PY="/data_huawei/gaohaizhen/network/saipn/model/ablation-d2/full.py"
CNLR_PY="${META_BASE}/count_CNLR.py"
ASSORT_PY="${META_BASE}/plot/associate/assortativity.py"

# Best params (keep consistent with full_best_meta)
TH="${TH:-0.70}"
DECAY="${DECAY:-336.0}"
WIN="${WIN:-1008.0}"
RESAMPLE="${RESAMPLE:-D}"
METRIC_INTERVAL="${METRIC_INTERVAL:-7}"

# Optional date range for assortativity plots
ASSORT_START_DATE="${ASSORT_START_DATE:-2021-10-01}"
ASSORT_END_DATE="${ASSORT_END_DATE:-2022-07-31}"

# Output suffix to avoid overwriting existing runs (override via: sbatch --export=ALL,OUT_SUFFIX=xxx)
OUT_SUFFIX="${OUT_SUFFIX:-v2}"

mkdir -p "${OUT_BASE}"

ensure_neutral_sentiment() {
  if [[ -f "${SENTI_NEUTRAL_FILE}" ]]; then
    return 0
  fi
  "${PY_BIN}" - <<'PY'
import pandas as pd
src="/data_huawei/gaohaizhen/network/saipn/model/metaverse/embeddin/final_with_sentiment2.csv"
dst="/data_huawei/gaohaizhen/network/saipn/model/metaverse/embeddin/final_with_sentiment_neutral.csv"
df=pd.read_csv(src)
df["positive_probability"]=0.0
df["negative_probability"]=0.0
df["neutral_probability"]=1.0
df.to_csv(dst,index=False)
print("[saved]", dst)
PY
}

run_full() {
  local name="$1"
  local decay="$2"
  local vec_file="$3"
  local senti_file="$4"

  local out_dir="${OUT_BASE}/${name}_${OUT_SUFFIX}"
  mkdir -p "${out_dir}"

  echo "==> [RERUN META] ${name}_${OUT_SUFFIX}: th=${TH}, decay=${decay}h, win=${WIN}h"
  "${PY_BIN}" "${FULL_PY}" \
    --resample "${RESAMPLE}" \
    --decay-unit-hours "${decay}" \
    --delete-after-hours "${WIN}" \
    --score-threshold "${TH}" \
    --data-file "${DATA}" \
    --vector-file "${vec_file}" \
    --sentiment-file "${senti_file}" \
    --output-dir "${out_dir}" \
    --metric-interval "${METRIC_INTERVAL}"

  if [[ ! -d "${out_dir}/snapshots" ]]; then
    echo "[Error] snapshots not found after run: ${out_dir}/snapshots" >&2
    exit 1
  fi

  echo "==> [CNLR indegree] ${name}_${OUT_SUFFIX} vs $(basename "${EXPLICIT_DIR}")"
  # metaverse/count_CNLR.py writes outputs under its own script directory (META_BASE),
  # not the current working directory. Clean both locations to avoid stale artifacts.
  rm -f cnlr_summary.csv cnlr_detailed.csv
  rm -f "${META_BASE}/cnlr_summary.csv" "${META_BASE}/cnlr_detailed.csv"
  "${PY_BIN}" "${CNLR_PY}" \
    --explicit-dir "${EXPLICIT_DIR}" \
    --implicit-dir "${out_dir}" \
    --metric indegree || true

  # Prefer outputs from META_BASE (authoritative), fallback to CWD.
  if [[ -f "${META_BASE}/cnlr_summary.csv" ]]; then
    mv "${META_BASE}/cnlr_summary.csv" "${out_dir}/cnlr_indegree_summary.csv"
    echo "  [saved] ${out_dir}/cnlr_indegree_summary.csv"
  elif [[ -f cnlr_summary.csv ]]; then
    mv cnlr_summary.csv "${out_dir}/cnlr_indegree_summary.csv"
    echo "  [saved] ${out_dir}/cnlr_indegree_summary.csv"
  else
    echo "  [warn] cnlr_summary.csv not produced; skip saving" >&2
  fi

  if [[ -f "${META_BASE}/cnlr_detailed.csv" ]]; then
    mv "${META_BASE}/cnlr_detailed.csv" "${out_dir}/cnlr_indegree_detailed.csv"
    echo "  [saved] ${out_dir}/cnlr_indegree_detailed.csv"
  elif [[ -f cnlr_detailed.csv ]]; then
    mv cnlr_detailed.csv "${out_dir}/cnlr_indegree_detailed.csv"
    echo "  [saved] ${out_dir}/cnlr_indegree_detailed.csv"
  else
    echo "  [warn] cnlr_detailed.csv not produced; skip saving" >&2
  fi

  echo "==> [ASSORTATIVITY snapshot] ${name}_${OUT_SUFFIX}"
  "${PY_BIN}" "${ASSORT_PY}" \
    --implicit-dir "${out_dir}/snapshots" \
    --explicit-dir "${EXPLICIT_DIR}/snapshots" \
    --output-dir "${out_dir}" \
    --start-date "${ASSORT_START_DATE}" \
    --end-date "${ASSORT_END_DATE}"

  echo "[Done] ${out_dir}"
  echo
}

# --- sanity checks ---
if [[ ! -d "${EXPLICIT_DIR}/snapshots" ]]; then
  echo "[Error] explicit snapshots dir not found: ${EXPLICIT_DIR}/snapshots" >&2
  exit 1
fi

ensure_neutral_sentiment

# 1) no_tags: swap vector file
run_full "ablation_no_tags_meta" "${DECAY}" "${VEC_NO_TAGS}" "${SENTI_FILE}"

# 2) no_time_decay: set huge decay
run_full "ablation_no_time_decay_meta" "${NO_TIME_DECAY_HOURS:-1000000000}" "${VEC_TAGS}" "${SENTI_FILE}"

# 3) no_sentiment: use neutral sentiment file
run_full "ablation_no_sentiment_meta" "${DECAY}" "${VEC_TAGS}" "${SENTI_NEUTRAL_FILE}"

# Optional: write a small summary CSV (index_gpu GLOBAL + CNLR median if exists)
SUMMARY_OUT="${OUT_BASE}/ablation_rerun_summary_${OUT_SUFFIX}.csv"
"${PY_BIN}" - <<PY
import pandas as pd
from pathlib import Path

out_base = Path("${OUT_BASE}")
runs = [
  ("ablation_no_tags_meta_${OUT_SUFFIX}", out_base / "ablation_no_tags_meta_${OUT_SUFFIX}"),
  ("ablation_no_time_decay_meta_${OUT_SUFFIX}", out_base / "ablation_no_time_decay_meta_${OUT_SUFFIX}"),
  ("ablation_no_sentiment_meta_${OUT_SUFFIX}", out_base / "ablation_no_sentiment_meta_${OUT_SUFFIX}"),
]

rows=[]
for name, d in runs:
  idx = d / "index_gpu.csv"
  if not idx.exists():
    continue
  df = pd.read_csv(idx)
  df["Time"] = df["Time"].astype(str)
  g = df[df["Time"].str.startswith("GLOBAL")].tail(1)
  if g.empty:
    continue
  g = g.iloc[0]
  cnlr_median = ""
  cnlr_sum = d / "cnlr_indegree_summary.csv"
  if cnlr_sum.exists():
    try:
      s = pd.read_csv(cnlr_sum)
      if not s.empty and "median_cnlr" in s.columns:
        cnlr_median = float(s.loc[0, "median_cnlr"])
    except Exception:
      pass

  rows.append({
    "Experiment": name,
    "Edge": float(g["Edges"]),
    "AvgPR": float(g["AvgPageRank"]),
    "ASS": float(g["Assortativity"]),
    "Mod": float(g["Modularity"]),
    "DCPRR": float(g["DCPRR"]),
    "Comp": float(g["CompIntensity"]),
    "Prop": float(g["PropScope"]),
    "Coll": float(g["CollabIntensity"]),
    "CNLR_index_global": float(g["CNLR"]),
    "CNLR_indegree_median": cnlr_median,
  })

out = pd.DataFrame(rows)
out.to_csv("${SUMMARY_OUT}", index=False)
print("[saved] ${SUMMARY_OUT}")
PY

echo "[All Done] rerun ablations with snapshots + CNLR + assortativity. Summary: ${SUMMARY_OUT}"
