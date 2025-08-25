#!/bin/bash
#SBATCH --job-name=gender-labeling-wave1      # Name of the job
#SBATCH --cpus-per-task=10
#SBATCH --mem=50GB
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/surprisingly_popular/output/job_%a.out
#SBATCH --error=slurm_logs/surprisingly_popular/error/job_%a.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100_80gb"
#SBATCH --array=0-14

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
  # Google Gemma Models
  "google/gemma-3-270m-it"
  "google/gemma-3-1b-it"
  "google/gemma-3-4b-it"
  "google/gemma-3-12b-it"
  # Mistral Models
  "ministral/Ministral-3b-instruct"
  "mistralai/Ministral-8B-Instruct-2410"
)

source /lustre/groups/eml/workspace/lgirrbach/miniforge3/etc/profile.d/conda.sh
conda activate vllm

python knowledge_distillation/surprisingly_popular.py \
    --model ${MODELS[$SLURM_ARRAY_TASK_ID]} \
    --truthfulqa_json knowledge_distillation/data/truthfulqa_mc.json
