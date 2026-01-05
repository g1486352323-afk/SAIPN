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
export HF_ENDPOINT=https://hf-mirror.com

# 路径配置
DATA="/data_huawei/gaohaizhen/network/saipn/model/compare_plot/meta/final_dataset_std.csv"
EMB_DIR="/data_huawei/gaohaizhen/network/saipn/model/metaverse/embeddin"
OUTPUT_BASE="/data_huawei/gaohaizhen/network/saipn/model/metaverse/output"
mkdir -p "${EMB_DIR}" "${OUTPUT_BASE}"

echo "============================================================"
echo "Metaverse Complete Pipeline"
echo "============================================================"
echo "Data: ${DATA}"
echo "Output: ${OUTPUT_BASE}"
echo ""

# ==================== STEP 1: 生成情感和向量 ====================
echo "============================================================"
echo "STEP 1: Generate Sentiment + Vectors"
echo "============================================================"

# 1.1 情感分数
if [ ! -f "${EMB_DIR}/final_with_sentiment.csv" ]; then
    echo "[1.1] Generating sentiment scores..."
    python /data_huawei/gaohaizhen/network/saipn/model/metaverse/gen_sentiment_chinese.py \
        --input "$DATA" \
        --output "${EMB_DIR}/final_with_sentiment.csv" \
        --text-col "raw_value.full_text" \
        --id-col "raw_value.id_str" \
        --batch 64 \
        --device cuda
else
    echo "[1.1] Sentiment file exists, skipping..."
fi

# 1.2 Tags 向量
if [ ! -f "${EMB_DIR}/output_vectors.txt" ]; then
    echo "[1.2] Generating Tags vectors..."
    python /data_huawei/gaohaizhen/network/saipn/model/ablation/gen_vectors_tags.py \
        --input "$DATA" \
        --output "${EMB_DIR}/output_vectors.txt" \
        --text-col "Tags" \
        --id-col "raw_value.id_str" \
        --model "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" \
        --batch 128 \
        --device cuda
else
    echo "[1.2] Tags vectors exist, skipping..."
fi

# 1.3 无Tags向量
if [ ! -f "${EMB_DIR}/output_vectors_no_tags.txt" ]; then
    echo "[1.3] Generating No-Tags vectors..."
    python /data_huawei/gaohaizhen/network/saipn/model/metaverse/gen_vectors_no_tags_chinese.py \
        --input "$DATA" \
        --output "${EMB_DIR}/output_vectors_no_tags.txt" \
        --text-col "raw_value.full_text" \
        --id-col "raw_value.id_str" \
        --model "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" \
        --batch 128 \
        --device cuda
else
    echo "[1.3] No-Tags vectors exist, skipping..."
fi

# ==================== STEP 2: 参数敏感性实验 ====================
echo ""
echo "============================================================"
echo "STEP 2: Parameter Sensitivity Analysis"
echo "============================================================"

DECAY=48.0
DELETE=72.0
RESAMPLE="H"
METRIC_INTERVAL=6

# 2.1 阈值敏感性
echo "[2.1] Testing Threshold Sensitivity..."
for TH in 0.60 0.65 0.70 0.75 0.80; do
    OUT_DIR="${OUTPUT_BASE}/sensitivity_th_${TH}"
    if [ ! -f "${OUT_DIR}/index_gpu.csv" ]; then
        echo "  Running Threshold=${TH}..."
        python /data_huawei/gaohaizhen/network/saipn/model/PHEME/full_fast.py \
            --data-file "$DATA" \
            --vector-file "${EMB_DIR}/output_vectors.txt" \
            --sentiment-file "${EMB_DIR}/final_with_sentiment.csv" \
            --output-dir "${OUT_DIR}" \
            --score-threshold $TH \
            --decay-unit-hours $DECAY \
            --delete-after-hours $DELETE \
            --resample $RESAMPLE \
            --metric-interval $METRIC_INTERVAL \
            --force
    else
        echo "  Threshold=${TH} exists, skipping..."
    fi
done

# 2.2 衰减敏感性
echo "[2.2] Testing Decay Sensitivity..."
for DC in 24.0 48.0 72.0 96.0; do
    OUT_DIR="${OUTPUT_BASE}/sensitivity_decay_${DC}"
    if [ ! -f "${OUT_DIR}/index_gpu.csv" ]; then
        echo "  Running Decay=${DC}..."
        python /data_huawei/gaohaizhen/network/saipn/model/PHEME/full_fast.py \
            --data-file "$DATA" \
            --vector-file "${EMB_DIR}/output_vectors.txt" \
            --sentiment-file "${EMB_DIR}/final_with_sentiment.csv" \
            --output-dir "${OUT_DIR}" \
            --score-threshold 0.70 \
            --decay-unit-hours $DC \
            --delete-after-hours $DELETE \
            --resample $RESAMPLE \
            --metric-interval $METRIC_INTERVAL \
            --force
    else
        echo "  Decay=${DC} exists, skipping..."
    fi
done

# 2.3 窗口敏感性
echo "[2.3] Testing Window Sensitivity..."
for WIN in 48.0 72.0 96.0 120.0; do
    OUT_DIR="${OUTPUT_BASE}/sensitivity_win_${WIN}"
    if [ ! -f "${OUT_DIR}/index_gpu.csv" ]; then
        echo "  Running Window=${WIN}..."
        python /data_huawei/gaohaizhen/network/saipn/model/PHEME/full_fast.py \
            --data-file "$DATA" \
            --vector-file "${EMB_DIR}/output_vectors.txt" \
            --sentiment-file "${EMB_DIR}/final_with_sentiment.csv" \
            --output-dir "${OUT_DIR}" \
            --score-threshold 0.70 \
            --decay-unit-hours $DECAY \
            --delete-after-hours $WIN \
            --resample $RESAMPLE \
            --metric-interval $METRIC_INTERVAL \
            --force
    else
        echo "  Window=${WIN} exists, skipping..."
    fi
done

# ==================== STEP 3: 汇总结果 ====================
echo ""
echo "============================================================"
echo "STEP 3: Summary"
echo "============================================================"

echo "Extracting GLOBAL_AVERAGE from all experiments..."
echo "Experiment,Nodes,Edges,PageRank,Assortativity,Modularity,DCPRR,CNLR" > "${OUTPUT_BASE}/sensitivity_summary.csv"

for f in ${OUTPUT_BASE}/sensitivity_*/index_gpu.csv; do
    if [ -f "$f" ]; then
        exp_name=$(dirname "$f" | xargs basename)
        avg_line=$(tail -1 "$f")
        echo "${exp_name},${avg_line}" | sed 's/GLOBAL_AVERAGE,//' >> "${OUTPUT_BASE}/sensitivity_summary.csv"
    fi
done

echo ""
echo "============================================================"
echo "Pipeline Complete!"
echo "============================================================"
echo "Results:"
echo "  - Embeddings: ${EMB_DIR}/"
echo "  - Experiments: ${OUTPUT_BASE}/sensitivity_*/"
echo "  - Summary: ${OUTPUT_BASE}/sensitivity_summary.csv"
