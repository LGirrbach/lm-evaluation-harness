#!/bin/bash
#SBATCH --job-name=score_answers      # Name of the job
#SBATCH --cpus-per-task=10
#SBATCH --mem=50GB
#SBATCH --time=10:00:00
#SBATCH --output=slurm_logs/score_answers/output/job_%a.out
#SBATCH --error=slurm_logs/score_answers/error/job_%a.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100_80gb|a100_80gb"
#SBATCH --array=0-16

# List of HF model names (3B–20B range)
MODELS=(
  # Qwen3 Models
  "Qwen/Qwen3-8B"
  "Qwen/Qwen3-14B"
  "Qwen/Qwen3-32B"
  # Llama Models
  "meta-llama/Llama-3.1-8B-Instruct"
  # Phi Models
  "microsoft/phi-4"
  # Other Models
  "inclusionAI/Ling-lite-1.5"
  "01-ai/Yi-1.5-34B-Chat"
  "internlm/internlm3-8b-instruct"
  "HuggingFaceH4/zephyr-7b-beta"
  "LGAI-EXAONE/EXAONE-4.0.1-32B"
  "baidu/ERNIE-4.5-21B-A3B-PT"
  "moonshotai/Moonlight-16B-A3B"
  "zai-org/GLM-4-9B-0414"
  "zai-org/GLM-4-32B-0414"
  "allenai/OLMo-2-0325-32B-Instruct"
  "allenai/Llama-3.1-Tulu-3.1-8B"
  "inclusionAI/GroveMoE-Inst"
)

source /lustre/groups/eml/workspace/lgirrbach/miniforge3/etc/profile.d/conda.sh
conda activate vllm

DATASETS=("bbh" "mmlu" "medmcqa" "commonsenseqa" "openbookqa")

for dataset in "${DATASETS[@]}"; do
    python knowledge_distillation/score_answers.py \
        --model ${MODELS[$SLURM_ARRAY_TASK_ID]} \
        --dataset "knowledge_distillation/data/${dataset}_standard.json" \
        --output_dir "results/score_answers/${dataset}/"
done

  