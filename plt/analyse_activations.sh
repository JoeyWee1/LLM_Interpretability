#!/bin/bash
#SBATCH -J my_gpu_job
#SBATCH -A MPHIL-DIS-SL2-GPU # GPU billing account
#SBATCH -p ampere # GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1 # Request 1 GPU
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/zyw26/rds/hpc-work/logs/%j.log
source /home/zyw26/Thesis/ThesisVenv/bin/activate
python /home/zyw26/Thesis/LLM_Interpretability/analyse_activations.py
