#!/bin/bash
#SBATCH --job-name=meta_sensitivity
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/data_huawei/gaohaizhen/network/saipn/model/metaverse/logs/%x_%j.out
#SBATCH --error=/data_huawei/gaohaizhen/network/saipn/model/metaverse/logs/%x_%j.err
#SBATCH --chdir=/data_huawei/gaohaizhen/network/saipn/model/metaverse

set -euxo pipefail
mkdir -p /data_huawei/gaohaizhen/network/saipn/model/metaverse/logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate network

# 路径配置
DATA="/data_huawei/gaohaizhen/network/saipn/model/compare_plot/meta/final_dataset_std.csv"
EMB_DIR="/data_huawei/gaohaizhen/network/saipn/model/metaverse/embeddin"
OUTPUT_BASE="/data_huawei/gaohaizhen/network/saipn/model/metaverse/output"

# 固定参数
DECAY=48.0
DELETE=72.0
RESAMPLE="H"
METRIC_INTERVAL=6

echo "=========================================="
echo "Metaverse: Parameter Sensitivity Analysis"
echo "=========================================="

# 参数敏感性: 阈值 (Threshold)
echo ""
echo "=== Testing Threshold Sensitivity ==="
for TH in 0.60 0.65 0.70 0.75 0.80; do
    echo "[Threshold=${TH}] Running..."
    python /data_huawei/gaohaizhen/network/saipn/model/PHEME/full_fast.py \
        --data-file "$DATA" \
        --vector-file "${EMB_DIR}/output_vectors.txt" \
        --sentiment-file "${EMB_DIR}/final_with_sentiment.csv" \
        --output-dir "${OUTPUT_BASE}/sensitivity_th_${TH}" \
        --score-threshold $TH \
        --decay-unit-hours $DECAY \
        --delete-after-hours $DELETE \
        --resample $RESAMPLE \
        --metric-interval $METRIC_INTERVAL \
        --force
done

# 参数敏感性: 时间衰减 (Decay)
echo ""
echo "=== Testing Decay Sensitivity ==="
for DC in 24.0 48.0 72.0 96.0; do
    echo "[Decay=${DC}] Running..."
    python /data_huawei/gaohaizhen/network/saipn/model/PHEME/full_fast.py \
        --data-file "$DATA" \
        --vector-file "${EMB_DIR}/output_vectors.txt" \
        --sentiment-file "${EMB_DIR}/final_with_sentiment.csv" \
        --output-dir "${OUTPUT_BASE}/sensitivity_decay_${DC}" \
        --score-threshold 0.70 \
        --decay-unit-hours $DC \
        --delete-after-hours $DELETE \
        --resample $RESAMPLE \
        --metric-interval $METRIC_INTERVAL \
        --force
done

# 参数敏感性: 窗口大小 (Delete After)
echo ""
echo "=== Testing Window Sensitivity ==="
for WIN in 48.0 72.0 96.0 120.0; do
    echo "[Window=${WIN}] Running..."
    python /data_huawei/gaohaizhen/network/saipn/model/PHEME/full_fast.py \
        --data-file "$DATA" \
        --vector-file "${EMB_DIR}/output_vectors.txt" \
        --sentiment-file "${EMB_DIR}/final_with_sentiment.csv" \
        --output-dir "${OUTPUT_BASE}/sensitivity_win_${WIN}" \
        --score-threshold 0.70 \
        --decay-unit-hours $DECAY \
        --delete-after-hours $WIN \
        --resample $RESAMPLE \
        --metric-interval $METRIC_INTERVAL \
        --force
done

echo ""
echo "=========================================="
echo "Sensitivity Analysis Complete!"
echo "=========================================="
echo "Results saved in ${OUTPUT_BASE}/sensitivity_*"
echo ""
echo "To find best parameters, compare GLOBAL_AVERAGE in each index_gpu.csv"
