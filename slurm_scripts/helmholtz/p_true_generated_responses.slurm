#!/bin/bash
#SBATCH --job-name=p_true_generated_responses      # Name of the job
#SBATCH --cpus-per-task=10
#SBATCH --mem=50GB
#SBATCH --time=23:00:00
#SBATCH --output=slurm_logs/p_true_generated_responses/output/job_%a.out
#SBATCH --error=slurm_logs/p_true_generated_responses/error/job_%a.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100_80gb|a100_80gb"
#SBATCH --array=2,8

# List of HF model names
MODELS=(
  "Qwen/Qwen3-8B"
  "Qwen/Qwen2.5-7B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
  "ibm-granite/granite-3.3-8b-instruct"
  "internlm/internlm3-8b-instruct"
  "zai-org/GLM-4-9B-0414"
  "google/gemma-2-9b-it"
  "HuggingFaceH4/zephyr-7b-beta"
  "01-ai/Yi-1.5-9B-Chat"
  "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
)

source /lustre/groups/eml/workspace/lgirrbach/miniforge3/etc/profile.d/conda.sh
conda activate vllm

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
#Cut off the model name after the last /
MODEL_NAME=$(echo ${MODEL} | rev | cut -d'/' -f1 | rev)

python knowledge_distillation/p_true_generated_responses.py \
    --input "results/generate_responses/${MODEL_NAME}.json" \
    --output "results/p_true_generated_responses/${MODEL_NAME}.csv" \
    --model "microsoft/phi-4"
