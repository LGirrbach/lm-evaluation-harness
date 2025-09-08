#!/bin/bash
#SBATCH --job-name=generate_responses      # Name of the job
#SBATCH --cpus-per-task=10
#SBATCH --mem=50GB
#SBATCH --time=23:00:00
#SBATCH --output=slurm_logs/generate_responses/output/job_%a.out
#SBATCH --error=slurm_logs/generate_responses/error/job_%a.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100_80gb"
#SBATCH --array=0-22

# List of HF model names
MODELS=(
  "Qwen/Qwen3-4B"
  "Qwen/Qwen3-8B"
  "Qwen/Qwen3-14B"
  "Qwen/Qwen2.5-7B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
  "ibm-granite/granite-3.3-8b-instruct"
  "internlm/internlm3-8b-instruct"
  "zai-org/GLM-4-9B-0414"
  "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
  "teknium/OpenHermes-2.5-Mistral-7B"
  "NousResearch/Hermes-4-14B"
  "tiiuae/Falcon3-10B-Instruct"
  "internlm/internlm2_5-20b-chat"
  "arcee-ai/AFM-4.5B"
  "deepseek-ai/DeepSeek-Prover-V2-7B"
  "deepseek-ai/deepseek-math-7b-instruct"
  "Qwen/Qwen2.5-Coder-7B-Instruct"
  "Qwen/Qwen2.5-Coder-14B-Instruct"
  "microsoft/phi-4-mini-instruct"
  "microsoft/phi-4"
  "inclusionAI/Ling-lite-1.5"
  "baidu/ERNIE-4.5-21B-A3B-PT"
)

source /lustre/groups/eml/workspace/lgirrbach/miniforge3/etc/profile.d/conda.sh
conda activate vllm

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
#Cut off the model name after the last /
MODEL_NAME=$(echo ${MODEL} | rev | cut -d'/' -f1 | rev)

python knowledge_distillation/generate_responses.py \
    --model ${MODEL} \
    --dataset "allenai/llama-3.1-tulu-3-8b-preference-mixture" \
    --split train \
    --output "results/generate_responses/tulu/${MODEL_NAME}.json" \
    --max-tokens 2048 \
    --temperature 0.7 \
    --top_p 0.9
