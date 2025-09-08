#!/bin/bash
#SBATCH --job-name=llm_as_judge      # Name of the job
#SBATCH --cpus-per-task=10
#SBATCH --mem=80GB
#SBATCH --time=20:00:00
#SBATCH --output=slurm_logs/llm_as_judge/output/job_%a.out
#SBATCH --error=slurm_logs/llm_as_judge/error/job_%a.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:2
#SBATCH --constraint="h100_80gb"
#SBATCH --array=0-77


MODELS=(
  "microsoft/phi-4"
  "inclusionAI/Ling-lite-1.5"
  "internlm/internlm2_5-20b-chat"
  "baidu/ERNIE-4.5-21B-A3B-PT"
  "Qwen/Qwen3-30B-A3B-Instruct-2507"
  "Qwen/Qwen3-32B"
  "baichuan-inc/Baichuan-M2-32B"
  "CohereLabs/c4ai-command-r-08-2024"
  "LGAI-EXAONE/EXAONE-4.0.1-32B"
  "zai-org/GLM-4-32B-0414"
  "microsoft/Phi-3.5-MoE-instruct"
  "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8"
  "nvidia/Llama-3.3-70B-Instruct-FP8"
)

DATASETS=("allenai/llama-3.1-tulu-3-8b-preference-mixture" "lmarena-ai/arena-human-preference-55k")

JUDGE_MODES=("binary" "rubrics" "holistic")

source /lustre/groups/eml/workspace/lgirrbach/miniforge3/etc/profile.d/conda.sh
conda activate vllm

# Calculate indices for model, dataset, and judge mode
# Total combinations: 13 models * 2 datasets * 3 judge modes = 78 combinations (0-77)
NUM_MODELS=${#MODELS[@]}
NUM_DATASETS=${#DATASETS[@]}
NUM_JUDGE_MODES=${#JUDGE_MODES[@]}

MODEL_IDX=$((SLURM_ARRAY_TASK_ID / (NUM_DATASETS * NUM_JUDGE_MODES)))
DATASET_IDX=$(((SLURM_ARRAY_TASK_ID % (NUM_DATASETS * NUM_JUDGE_MODES)) / NUM_JUDGE_MODES))
JUDGE_MODE_IDX=$((SLURM_ARRAY_TASK_ID % NUM_JUDGE_MODES))

MODEL=${MODELS[$MODEL_IDX]}
DATASET=${DATASETS[$DATASET_IDX]}
JUDGE_MODE=${JUDGE_MODES[$JUDGE_MODE_IDX]}

#Cut off the model name after the last /
MODEL_NAME=$(echo ${MODEL} | rev | cut -d'/' -f1 | rev)
DATASET_NAME=$(echo ${DATASET} | rev | cut -d'/' -f1 | rev)

python knowledge_distillation/p_true_preference.py \
    --dataset $DATASET \
    --output "results/llm_as_judge/${MODEL_NAME}/${JUDGE_MODE}/${DATASET_NAME}.csv" \
    --model ${MODEL} \
    --logprobs-k 15 \
    --max-examples 20000 \
    --include-ties \
    --tensor-parallel-size 2 \
    --judge-mode $JUDGE_MODE