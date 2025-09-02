#!/bin/bash
#SBATCH -A NAISS2025-5-150 -p alvis
#SBATCH --time=0-03:00:00
#SBATCH -N 1 --gpus-per-node V100:1
#SBATCH --output=/dev/null
# Max parameter values
# gpu 4, cpu 40, time 2-00:00:00, 117 GB

ml PyTorch-Geometric/2.5.0-foss-2023a-PyTorch-2.1.2-CUDA-12.1.1
ml Transformers/4.39.3-gfbf-2023a
ml torchvision/0.16.0-foss-2023a-CUDA-12.1.1
ml Seaborn/0.13.2-gfbf-2023a
timestamp=$(date +%y%m%d_%H%M)

python ./run_extraction_attack.py > ./outputs/script_logs/extraction_attack_${timestamp}.txt 2>&1