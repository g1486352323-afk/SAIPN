#!/bin/bash
#SBATCH --job-name=ablation_full
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=/data_huawei/gaohaizhen/network/saipn/model/ablation/logs/%x_%j.out
#SBATCH --error=/data_huawei/gaohaizhen/network/saipn/model/ablation/logs/%x_%j.err
#SBATCH --chdir=/data_huawei/gaohaizhen/network/saipn/model/ablation

# --- 0. 环境准备 ---
# 创建日志目录防止报错
mkdir -p logs

# 设置 Shell 严格模式：遇到错误立即退出，打印执行命令
set -euxo pipefail

export PYTHONUNBUFFERED=1
export TORCH_SHOW_CPP_STACKTRACES=1

echo "===== SLURM 作业信息 ====="
date
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPU: ${SLURM_JOB_GPUS}"
echo "=========================="

# 激活环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate network

# 检查环境
nvidia-smi
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import cugraph; print(f'CuGraph: {cugraph.__version__}')" || echo "Warning: CuGraph not found"
export HF_ENDPOINT=https://hf-mirror.com
# ================= 1. 变量定义 (集中管理路径) =================
BASE_DIR="/data_huawei/gaohaizhen/network/saipn/model/ablation"
DATA="${BASE_DIR}/charliehebdo_gemini_2_flash_output_fixed_from_cleaned.csv"
EMBED_DIR="${BASE_DIR}/embeddin"

# 确保输出目录存在
mkdir -p "$EMBED_DIR"

# 定义关键文件路径
VECTOR="${EMBED_DIR}/output_vectors.txt"               # 原始向量(需已存在)
VECTOR_NOTAG="${EMBED_DIR}/output_vectors_no_tags.txt" # 无Tag向量(待生成)
SENTI="${EMBED_DIR}/final_with_sentiment.csv"          # 情感文件(待生成)

# 定义实验通用参数 (小时级采样, 12小时衰减, 24小时生命周期, 0.7阈值)
COMMON_ARGS="--resample H --decay-unit-hours 12.0 --delete-after-hours 24.0 --score-threshold 0.7"

# ================= 2. 数据预处理阶段 =================

# echo ">>> [Step 1] 生成/检查 情感文件..."
# if [ ! -f "$SENTI" ]; then
#     python ${BASE_DIR}/gen_sentiment.py \
#         --input "$DATA" \
#         --output "$SENTI" \
#         --text-col "raw_value.text" \
#         --id-col "raw_value.id_str" \
#         --model "cardiffnlp/twitter-roberta-base-sentiment-latest" \
#         --batch 64 \
#         --device cuda
# else
#     echo "文件已存在，跳过: $SENTI"
# fi

# echo ">>> [Step 2] 生成/检查 无Tag向量文件..."
# if [ ! -f "$VECTOR_NOTAG" ]; then
#     # 注意：使用修正后的脚本，参数为 --input
#     python ${BASE_DIR}/gen_vectors_no_tags.py \
#         --input "$DATA" \
#         --output "$VECTOR_NOTAG" \
#         --batch-size 128 \

#         --device cuda

# else
#     echo "文件已存在，跳过: $VECTOR_NOTAG"
# fi

# echo ">>> [Step 2] 生成/检查 Tag向量文件 (修正版)..."
# # 注意：这里检查的是 VECTOR (最终目标)，不要检查 VECTOR_NOTAG
# if [ ! -f "$VECTOR" ]; then
#     python ${BASE_DIR}/gen_vectors.py \
#         --input "$DATA" \
#         --output "$VECTOR" \
#         --text-col Tags \
#         --device cuda
#         # ⬆️ 确保这里没有反斜杠，也没有 ' '
# else
#     echo "文件已存在，跳过: $VECTOR"
# fi
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