#!/bin/bash
#SBATCH --job-name=meta_sensitivity
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=/data_huawei/gaohaizhen/network/saipn/model/metaverse/logs/%x_%j.out
#SBATCH --error=/data_huawei/gaohaizhen/network/saipn/model/metaverse/logs/%x_%j.err
#SBATCH --chdir=/data_huawei/gaohaizhen/network/saipn/model/compare_plot/pheme

set -euxo pipefail

PY_BIN="/data_huawei/gaohaizhen/.conda/envs/network/bin/python"

MODE="${1:-full}"

META_BASE="/data_huawei/gaohaizhen/network/saipn/model/metaverse"
OUT_BASE="${META_BASE}/output"
DATA="/data_huawei/gaohaizhen/network/saipn/model/metaverse/final_dataset_std_with_tags_filled.csv"
VEC_TAGS="${META_BASE}/embeddin/output_vectors_tags.txt"
SENTI_FILE="${META_BASE}/embeddin/final_with_sentiment2.csv"
EXPLICIT_DIR="${META_BASE}/output/explicit_metaverse"

mkdir -p "${META_BASE}/logs" "${OUT_BASE}"

if [ "${MODE}" = "dcprr" ]; then
  "${PY_BIN}" "${META_BASE}/recompute_dcprr_penalized.py" \
    --output-base "${OUT_BASE}" \
    --run-glob "[ABC]_*" \
    --metric-interval 7 \
    --walk-length 100 \
    --num-walks 30 \
    --seed 42 \
    --community-resolution 1.0 \
    --min-community-size-detect 4 \
    --max-start-nodes 200 \
    --k 1.0 \
    --m 6.0 \
    --out-name index_gpu_dcprr_penalized.csv

  "${PY_BIN}" "${META_BASE}/summarize_sensitivity_penalized.py" \
    --output-base "${OUT_BASE}" \
    --run-glob "[ABC]_*" \
    --original-name index_gpu.csv \
    --penalized-name index_gpu_dcprr_penalized.csv \
    --out-name sensitivity_summary_penalized.csv
  exit 0
fi

run_one() {
  local name="$1"
  local th="$2"
  local decay="$3"
  local win="$4"

  local out_dir="${OUT_BASE}/${name}"
  mkdir -p "${out_dir}"

  echo "==> [Implicit-META] ${name}: th=${th}, decay=${decay}h, win=${win}h"
  "${PY_BIN}" /data_huawei/gaohaizhen/network/saipn/model/ablation-d2/full.py \
    --resample D \
    --decay-unit-hours "${decay}" \
    --delete-after-hours "${win}" \
    --score-threshold "${th}" \
    --data-file "${DATA}" \
    --vector-file "${VEC_TAGS}" \
    --sentiment-file "${SENTI_FILE}" \
    --output-dir "${out_dir}" \
    --metric-interval 7

  echo "==> [CNLR indegree META] ${name}"
  "${PY_BIN}" /data_huawei/gaohaizhen/network/saipn/model/compare_plot/pheme/count_CNLR.py \
    --explicit-dir "${EXPLICIT_DIR}" \
    --implicit-dir "${out_dir}" \
    --metric indegree || true

  [ -f cnlr_summary.csv ] && mv cnlr_summary.csv "${out_dir}/cnlr_indegree_summary.csv"
  [ -f cnlr_detailed.csv ] && mv cnlr_detailed.csv "${out_dir}/cnlr_indegree_detailed.csv"
}

# === Group A: 阈值敏感性 (半衰期 15 天 = 360h, 窗口 30 天 = 720h) ===
run_one "A_Th_0.50_Win30d" 0.50 360.0 720.0
run_one "A_Th_0.55_Win30d" 0.55 360.0 720.0
run_one "A_Th_0.60_Win30d" 0.60 360.0 720.0
run_one "A_Th_0.65_Win30d" 0.65 360.0 720.0
run_one "A_Th_0.70_Win30d" 0.70 360.0 720.0
run_one "A_Th_0.75_Win30d" 0.75 360.0 720.0
run_one "A_Th_0.80_Win30d" 0.80 360.0 720.0
run_one "A_Th_0.85_Win30d" 0.85 360.0 720.0
run_one "A_Th_0.90_Win30d" 0.90 360.0 720.0
run_one "A_Th_0.95_Win30d" 0.95 360.0 720.0

# === Group B: 窗口规模敏感性 (7 天 -> 70 天, 复用 ablation-d2/run_sensitivity.py 的参数) ===
run_one "B_Win_07d_1wk" 0.70 84.0   168.0
run_one "B_Win_14d_2wk" 0.70 168.0  336.0
run_one "B_Win_21d_3wk" 0.70 252.0  504.0
run_one "B_Win_28d_4wk" 0.70 336.0  672.0
run_one "B_Win_35d_5wk" 0.70 420.0  840.0
run_one "B_Win_42d_6wk" 0.70 504.0  1008.0
run_one "B_Win_49d_7wk" 0.70 588.0  1176.0
run_one "B_Win_56d_8wk" 0.70 672.0  1344.0
run_one "B_Win_63d_9wk" 0.70 756.0  1512.0
run_one "B_Win_70d_10wk" 0.70 840.0 1680.0

# === Group C: 衰减速度敏感性 (固定窗口 60 天 = 1440h) ===
run_one "C_Decay_02d_Win60d" 0.70 48.0   1440.0
run_one "C_Decay_05d_Win60d" 0.70 120.0  1440.0
run_one "C_Decay_07d_Win60d" 0.70 168.0  1440.0
run_one "C_Decay_10d_Win60d" 0.70 240.0  1440.0
run_one "C_Decay_14d_Win60d" 0.70 336.0  1440.0
run_one "C_Decay_21d_Win60d" 0.70 504.0  1440.0
run_one "C_Decay_30d_Win60d" 0.70 720.0  1440.0
run_one "C_Decay_40d_Win60d" 0.70 960.0  1440.0
run_one "C_Decay_50d_Win60d" 0.70 1200.0 1440.0
run_one "C_Decay_60d_Win60d" 0.70 1440.0 1440.0

echo "All METAVERSE TAG-based sensitivity runs and CNLR(indegree) computations done."
