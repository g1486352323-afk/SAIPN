#!/bin/bash
#SBATCH --job-name=meta_finalize_v2
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
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
EXPLICIT_DIR="${OUT_BASE}/explicit_metaverse_best"
CNLR_PY="${META_BASE}/count_CNLR.py"

ASSORT_PY="${META_BASE}/plot/associate/assortativity.py"
ASSORT_START_DATE="${ASSORT_START_DATE:-2021-10-01}"
ASSORT_END_DATE="${ASSORT_END_DATE:-2022-07-31}"

RUNS=(
  "full_best_meta"
  "ablation_no_tags_meta_v2"
  "ablation_no_time_decay_meta_v2"
  "ablation_no_sentiment_meta_v2"
)

for r in "${RUNS[@]}"; do
  IMPL_DIR="${OUT_BASE}/${r}"
  rm -f "${META_BASE}/cnlr_summary.csv" "${META_BASE}/cnlr_detailed.csv"
  "${PY_BIN}" "${CNLR_PY}" \
    --explicit-dir "${EXPLICIT_DIR}" \
    --implicit-dir "${IMPL_DIR}" \
    --metric indegree

  if [[ -f "${META_BASE}/cnlr_summary.csv" ]]; then
    mv "${META_BASE}/cnlr_summary.csv" "${IMPL_DIR}/cnlr_indegree_summary.csv"
  fi
  if [[ -f "${META_BASE}/cnlr_detailed.csv" ]]; then
    mv "${META_BASE}/cnlr_detailed.csv" "${IMPL_DIR}/cnlr_indegree_detailed.csv"
  fi

  if [[ ! -f "${IMPL_DIR}/assortativity_implicit.csv" ]]; then
    if [[ ! -d "${IMPL_DIR}/snapshots" ]]; then
      echo "[Warn] snapshots not found for ${IMPL_DIR}, skip assortativity." >&2
    elif [[ ! -d "${EXPLICIT_DIR}/snapshots" ]]; then
      echo "[Warn] explicit snapshots not found for ${EXPLICIT_DIR}/snapshots, skip assortativity." >&2
    else
      "${PY_BIN}" "${ASSORT_PY}" \
        --implicit-dir "${IMPL_DIR}/snapshots" \
        --explicit-dir "${EXPLICIT_DIR}/snapshots" \
        --output-dir "${IMPL_DIR}" \
        --start-date "${ASSORT_START_DATE}" \
        --end-date "${ASSORT_END_DATE}"
    fi
  fi

done

FINAL_OUT="${OUT_BASE}/ablation_rerun_summary_v2_final.csv"
"${PY_BIN}" - <<PY
import pandas as pd
from pathlib import Path

out_base = Path("${OUT_BASE}")
runs = [
  ("full_best_meta", out_base/"full_best_meta"),
  ("ablation_no_tags_meta_v2", out_base/"ablation_no_tags_meta_v2"),
  ("ablation_no_time_decay_meta_v2", out_base/"ablation_no_time_decay_meta_v2"),
  ("ablation_no_sentiment_meta_v2", out_base/"ablation_no_sentiment_meta_v2"),
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

  cnlr_med = ""
  cnlr_overlap = ""
  cnlr_sum = d / "cnlr_indegree_summary.csv"
  if cnlr_sum.exists():
    s = pd.read_csv(cnlr_sum)
    if not s.empty:
      if "median_cnlr" in s.columns:
        cnlr_med = float(s.loc[0, "median_cnlr"])
      if "total_overlap_nodes" in s.columns:
        cnlr_overlap = int(s.loc[0, "total_overlap_nodes"])

  assort_med = ""
  assort_mean = ""
  assort_csv = d / "assortativity_implicit.csv"
  if assort_csv.exists():
    a = pd.read_csv(assort_csv)
    s = pd.to_numeric(a.get("degree_assortativity"), errors="coerce").dropna()
    if len(s):
      assort_med = float(s.median())
      assort_mean = float(s.mean())

  rows.append({
    "Experiment": name,
    "Edges_index_global": float(g["Edges"]),
    "AvgPageRank_index_global": float(g["AvgPageRank"]),
    "Assortativity_index_global": float(g["Assortativity"]),
    "Assortativity_snapshot_median": assort_med,
    "Assortativity_snapshot_mean": assort_mean,
    "Modularity_index_global": float(g["Modularity"]),
    "DCPRR_index_global": float(g["DCPRR"]),
    "CompIntensity_index_global": float(g["CompIntensity"]),
    "PropScope_index_global": float(g["PropScope"]),
    "CollabIntensity_index_global": float(g["CollabIntensity"]),
    "CNLR_index_global": float(g["CNLR"]),
    "CNLR_indegree_median": cnlr_med,
    "CNLR_overlap_nodes": cnlr_overlap,
  })

out = pd.DataFrame(rows)
out.to_csv("${FINAL_OUT}", index=False)
print("[saved] ${FINAL_OUT}")
PY

echo "[All Done] Final summary: ${FINAL_OUT}"
