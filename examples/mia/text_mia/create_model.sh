#!/bin/bash
#SBATCH -A NAISS2024-5-119 -p alvis
#SBATCH --time=1-00:00:00
#SBATCH --gpus-per-node A40:1
#SBATCH -o ./output_extraction_attack.txt
# Max parameter values
# gpu 4, cpu 40, time 2-00:00:00, 117 GB

ml PyTorch-Geometric/2.5.0-foss-2023a-PyTorch-2.1.2-CUDA-12.1.1
ml PDM/2.12.4-GCCcore-12.3.0
ml Transformers/4.39.3-gfbf-2023a
ml torchvision/0.16.0-foss-2023a-CUDA-12.1.1
ml Seaborn/0.13.2-gfbf-2023a
python ./create_target_model.py