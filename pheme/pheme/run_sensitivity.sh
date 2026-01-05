#!/bin/bash
#SBATCH --job-name=ablation_sensitivity
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

mkdir -p logs
set -euxo pipefail

export PYTHONUNBUFFERED=1
export TORCH_SHOW_CPP_STACKTRACES=1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate network

nvidia-smi
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import cugraph; print(f'CuGraph: {cugraph.__version__}')" || echo "Warning: CuGraph not found"

echo "===== SLURM 作业信息 ====="
date
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPU: ${SLURM_JOB_GPUS}"
echo "=========================="

python /data_huawei/gaohaizhen/network/saipn/model/ablation/run_sensitivity.py

echo "===== 任务完成 ====="
date


