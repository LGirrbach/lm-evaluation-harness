#!/bin/bash
#SBATCH --job-name=rate-generated-responses-phi4-ultrachat      # Name of the job
#SBATCH --cpus-per-task=10
#SBATCH --mem=80GB
#SBATCH --time=20:00:00
#SBATCH --output=slurm_logs/rate-generated-responses/phi4/ultrachat/output/job_%a.out
#SBATCH --error=slurm_logs/rate-generated-responses/phi4/ultrachat/error/job_%a.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100_80gb"
#SBATCH --array=02,14-20

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
  "NousResearch/Hermes-4-14B"
  "tiiuae/Falcon3-10B-Instruct"
  "arcee-ai/AFM-4.5B"
  "microsoft/phi-4-mini-instruct"
  "microsoft/phi-4"
  "inclusionAI/Ling-lite-1.5"
  "baidu/ERNIE-4.5-21B-A3B-PT"
  "Qwen/Qwen2.5-Coder-7B-Instruct"
  "Qwen/Qwen2.5-Coder-14B-Instruct"
  "teknium/OpenHermes-2.5-Mistral-7B"
  "internlm/internlm2_5-20b-chat"
)

source /lustre/groups/eml/workspace/lgirrbach/miniforge3/etc/profile.d/conda.sh
conda activate vllm

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
#Cut off the model name after the last /
MODEL_NAME=$(echo ${MODEL} | rev | cut -d'/' -f1 | rev)

python knowledge_distillation/rate_generated_responses.py \
    --response-json "results/generate_responses/ultrachat/${MODEL_NAME}.json" \
    --output "results/generate_responses/ratings/ultrachat/${MODEL_NAME}.csv" \
    --model "microsoft/phi-4" \
    --logprobs-k 15 \
    --judge-mode holistic