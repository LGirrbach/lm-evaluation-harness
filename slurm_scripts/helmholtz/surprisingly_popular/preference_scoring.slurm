#!/bin/bash
#SBATCH --job-name=preference_scoring      # Name of the job
#SBATCH --cpus-per-task=10
#SBATCH --mem=50GB
#SBATCH --time=10:00:00
#SBATCH --output=slurm_logs/preference_scoring/output/job_%a.out
#SBATCH --error=slurm_logs/preference_scoring/error/job_%a.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100_80gb|a100_80gb"
#SBATCH --array=0-16

# List of HF model names (3B–20B range)
MODELS=(
  # Qwen3 Models
  "Qwen/Qwen3-0.6B"
  "Qwen/Qwen3-1.7B"
  "Qwen/Qwen3-4B"
  "Qwen/Qwen3-8B"
  "Qwen/Qwen3-14B"
  # Llama Models
  "meta-llama/Llama-3.2-1B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
  # Phi Models
  "microsoft/phi-4"
  # Other Models
  "google/gemma-3-4b-it"
  "inclusionAI/Ling-lite-1.5"
  "internlm/internlm3-8b-instruct"
  "HuggingFaceH4/zephyr-7b-beta"
  "baidu/ERNIE-4.5-21B-A3B-PT"
  "moonshotai/Moonlight-16B-A3B"
  "zai-org/GLM-4-9B-0414"
  "allenai/Llama-3.1-Tulu-3.1-8B"
)

source /lustre/groups/eml/workspace/lgirrbach/miniforge3/etc/profile.d/conda.sh
conda activate vllm

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
#Cut off the model name after the last /
MODEL_NAME=$(echo ${MODEL} | rev | cut -d'/' -f1 | rev)

python knowledge_distillation/score_preference_data.py \
    --dataset "allenai/llama-3.1-tulu-3-8b-preference-mixture" \
    --output "results/preference_scoring/${MODEL_NAME}.csv" \
    --model ${MODEL}

python knowledge_distillation/p_true_preference.py \
    --dataset "allenai/llama-3.1-tulu-3-8b-preference-mixture" \
    --output "results/p_true_preference/${MODEL_NAME}.csv" \
    --model ${MODEL}