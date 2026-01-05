#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=h100                # <-- 【修正】分区名从 h100 改为 main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1                     # <-- 【简化】请求1个通用GPU资源
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=/data_huawei/gaohaizhen/network/saipn/model/ablation-d2/logs/%x_%j.out
#SBATCH --error=/data_huawei/gaohaizhen/network/saipn/model/ablation-d2/logs/%x_%j.err
#SBATCH --chdir=/data_huawei/gaohaizhen/network/saipn/model/ablation-d2

# --- 预创建日志目录 ---
mkdir -p logs

export PYTHONUNBUFFERED=1
export TORCH_SHOW_CPP_STACKTRACES=1

# 严格模式
set -euxo pipefail

echo "===== SLURM 环境检查 ====="
date
hostname
echo "作业ID: ${SLURM_JOB_ID}"
echo "作业名称: ${SLURM_JOB_NAME}"
echo "提交节点: ${SLURM_SUBMIT_HOST}"
echo "运行节点: ${SLURM_NODELIST}"
echo "申请的GPU: ${SLURM_JOB_GPUS}"
echo "=========================="

# 激活 Conda 环境
# 确保 conda 命令可用
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate network

echo "===== Python 和 CUDA 环境检查 ====="
echo "Python 路径: $(which python)"


# 检查NVIDIA驱动和GPU状态
echo "===== GPU 状态 (nvidia-smi) ====="
nvidia-smi
echo "==================================="

# 运行你的Python脚本
python -c "import cugraph; print('Success! Cugraph version:', cugraph.__version__)"
#HF_ENDPOINT=https://hf-mirror.com python gen_vectors.py --input /data_huawei/gaohaizhen/network/saipn/model/ablation-d2/final_with_sentiment.csv
#python gen_sentiment.py  --input /data_huawei/gaohaizhen/network/saipn/model/ablation-d2/final_with_sentiment.csv
#python gen_vectors.py  --input /data_huawei/gaohaizhen/network/saipn/model/ablation-d2/final_with_sentiment.csv
# python run_pipeline.py \
#   --input /data_huawei/gaohaizhen/network/saipn/model/charliehebdo_gemini_2_flash_output_fixed.csv \
#   --workdir outputs \
#   --theta 0.98 \
#   --t-max 30 \
#   --encoder bert \
#   --bert-model ./local_models/bert-base-uncased \
#   --senti-model ./local_models/cardiffnlp-twitter-roberta-base-sentiment \
#   --hf-local-only \
#   --threads 16 --batch-sent 256 --batch-sem 128 \
#   --use-cugraph
# python generate_timeseries.py
#python run_sensitivity.py


#HF_ENDPOINT=https://hf-mirror.com  python  /data_huawei/gaohaizhen/network/saipn/model/ablation-d2/gen_vectors_no_tags.py
HF_ENDPOINT=https://hf-mirror.com python /data_huawei/gaohaizhen/network/saipn/model/ablation-d2/gen_sentiment.py \
  --input /data_huawei/gaohaizhen/network/saipn/model/charliehebdo_gemini_2_flash_output_fixed.csv \
  --output /data_huawei/gaohaizhen/network/saipn/model/ablation-d2/embedding/final_with_sentiment2.csv
# 1. 运行 Tag 消融实验 (确保无Tag向量已生成)
# python ablation_no_sentiment.py \
#   --score-threshold 0.70 \
#   --delete-after-hours 720.0 \
#   --decay-unit-hours 360.0 \
#   --resample D

# python ablation_no_tags.py \
#   --score-threshold 0.70 \
#   --delete-after-hours 720.0 \
#   --decay-unit-hours 360.0 \
#   --resample D

# python ablation_no_time_decay.py \
#   --score-threshold 0.70 \
#   --delete-after-hours 720.0 \
#   --decay-unit-hours 360.0 \
#   --resample D


python /data_huawei/gaohaizhen/network/saipn/model/ablation-d2/full.py \
  --score-threshold 0.70 \
  --delete-after-hours 720.0 \
  --decay-unit-hours 360.0 \
  --resample D

python  /data_huawei/gaohaizhen/network/saipn/model/ablation-d2/summary_ablation.py

echo "===== 作业完成 ====="
#python /data_huawei/gaohaizhen/network/saipn/model/ablation/full.py --score-threshold 0.70
date
echo "运行用户: $(whoami 2>/dev/null || echo "lookup failed for UID ${UID}")"



export HF_ENDPOINT=https://hf-mirror.com
# ================= 1. 变量定义 (集中管理路径) =================
BASE_DIR="/data_huawei/gaohaizhen/network/saipn/model/ablation2"
DATA="/data_huawei/gaohaizhen/network/saipn/model/compare_plot/meta/final_dataset_std.csv"
EMBED_DIR="${BASE_DIR}/embedding"

# 确保输出目录存在
mkdir -p "$EMBED_DIR"

# 定义关键文件路径
VECTOR="${EMBED_DIR}/output_vectors.txt"               # 原始向量(需已存在)
VECTOR_NOTAG="${EMBED_DIR}/output_vectors_no_tags.txt" # 无Tag向量(待生成)
SENTI="${EMBED_DIR}/final_with_sentiment.csv"          # 情感文件(待生成)

# 定义实验通用参数 (小时级采样, 2小时衰减, 24小时生命周期, 0.7阈值)
COMMON_ARGS="--resample H --decay-unit-hours 2.0 --delete-after-hours 24.0 --score-threshold 0.7"

# ================= 2. 数据预处理阶段 =================

echo ">>> [Step 1] 生成/检查 情感文件..."
if [ ! -f "$SENTI" ]; then
    python ${BASE_DIR}/gen_sentiment.py \
        --input "$DATA" \
        --output "$SENTI" \
        --text-col "raw_value.text" \
        --id-col "raw_value.id_str" \
        --model "cardiffnlp/twitter-roberta-base-sentiment-latest" \
        --batch 64 \
        --device cuda
else
    echo "文件已存在，跳过: $SENTI"
fi

echo ">>> [Step 2] 生成/检查 无Tag向量文件..."
if [ ! -f "$VECTOR_NOTAG" ]; then
    # 注意：使用修正后的脚本，参数为 --input
    python ${BASE_DIR}/gen_vectors_no_tags.py \
        --input "$DATA" \
        --output "$VECTOR_NOTAG" \
        --batch-size 128 \

        --device cuda

else
    echo "文件已存在，跳过: $VECTOR_NOTAG"
fi

echo ">>> [Step 2] 生成/检查 Tag向量文件 (修正版)..."
# 注意：这里检查的是 VECTOR (最终目标)，不要检查 VECTOR_NOTAG
if [ ! -f "$VECTOR" ]; then
    python ${BASE_DIR}/gen_vectors.py \
        --input "$DATA" \
        --output "$VECTOR" \
        --text-col Tags \
        --device cuda
        # ⬆️ 确保这里没有反斜杠，也没有 ' '
else
    echo "文件已存在，跳过: $VECTOR"
fi
# ================= 3. 消融实验阶段 =================

# 注意：这里需要传入 --data-file 等具体路径参数，不能只传 H/2/24

# echo ">>> [Step 3.1] 运行消融实验: 无时间衰减 (No Time Decay)"
# python ${BASE_DIR}/ablation_no_time_decay.py \
#     $COMMON_ARGS \
#     --data-file "$DATA" \
#     --vector-file "$VECTOR" \
#     --sentiment-file "$SENTI"

echo ">>> [Step 3.2] 运行消融实验: 全量模型 (Full Model Baseline)"
# 加上 --force 确保结果是最新的
python ${BASE_DIR}/full.py \
    $COMMON_ARGS \
    --force \
    --data-file "$DATA" \
    --vector-file "$VECTOR" \
    --sentiment-file "$SENTI"

# 如果你需要跑另外两个消融实验，可以取消下面的注释：

# echo ">>> [Step 3.3] 运行消融实验: 无情感 (No Sentiment)"
# python ${BASE_DIR}/ablation_no_sentiment.py \
#     $COMMON_ARGS \
#     --data-file "$DATA" \
#     --vector-file "$VECTOR" \
#     --sentiment-file "$SENTI"

# echo ">>> [Step 3.4] 运行消融实验: 无标签 (No Tags)"
# python ${BASE_DIR}/ablation_no_tags.py \
#     $COMMON_ARGS \
#     --data-file "$DATA" \
#     --vector-file "$VECTOR_NOTAG" \
#     --sentiment-file "$SENTI"

echo "===== 所有任务执行完毕 ====="
date