#!/bin/bash
#SBATCH --job-name=model-evaluation           # Name of the job
#SBATCH --nodes=1                            # Request 1 nodes
#SBATCH --ntasks-per-node=1                  # Start only 1 task per node
#SBATCH --gres=gpu:1                         # Request 1 GPU
#SBATCH --cpus-per-task=8                    # Request 8 cpus per task
#SBATCH --mem=70GB                           # Request 70GB of cpu memory per node
#SBATCH --time=12:00:00                      # Time limit hrs:min:sec (adjust as needed)
#SBATCH --output=slurm_logs/evaluation/output/evaluate_models_%A_%a.out  # Output file (%A=array job ID, %a=array task ID)
#SBATCH --error=slurm_logs/evaluation/error/evaluate_models_%A_%a.err    # Error log file
#SBATCH --partition=lrz-hgx-h100-94x4,lrz-hgx-a100-80x4
#SBATCH --exclude=lrz-hgx-h100-027
#SBATCH --array=0-19                         # Array indices for 20 models

set -euo pipefail

# Directory to store results
RESULTS_DIR="./results_scratch"

# List of HF model names (3B–20B range)
MODELS=(
  # Qwen3 Models

  Qwen/Qwen3-0.6B
  "openai/gpt-oss-20b"
  "meta-llama/Llama-3.1-8B-Instruct"
  "google/gemma-3-1b-it"
  "Qwen/Qwen3-4B"
  "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
  "meta-llama/Llama-3.1-8B"
  "microsoft/Phi-3-mini-4k-instruct"
  "Qwen/Qwen3-8B"
  "mistralai/Mistral-7B-Instruct-v0.2"
  "Qwen/Qwen3-14B"
  "microsoft/phi-4"
  "microsoft/Phi-4-mini-flash-reasoning"
  "google/gemma-7b-it"
)

# Popular multiple-choice tasks
TASKS="arc_easy,arc_challenge,hellaswag,winogrande,boolq,piqa,openbookqa,commonsense_qa,headqa,prost"

# Number of few-shot examples
FEWSHOT=5

# Get the model for this array task
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
SAFE_MODEL_NAME=$(echo "$MODEL" | tr '/:' '_')
OUT_FILE="$RESULTS_DIR/${SAFE_MODEL_NAME}/results.json"
LOG_FILE="$RESULTS_DIR/${SAFE_MODEL_NAME}/samples.jsonl"

echo "Evaluating $MODEL (Array task $SLURM_ARRAY_TASK_ID)..."
echo "Output will be saved to: $OUT_FILE"
echo "Log samples will be saved to: $LOG_FILE"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

python -m lm_eval \
    --model hf \
    --model_args pretrained="$MODEL" \
    --tasks "$TASKS" \
    --num_fewshot "$FEWSHOT" \
    --batch_size auto \
    --output_path "$OUT_FILE" \
    --log_samples \
    --log_samples_path "$LOG_FILE" \
    --use_cache  # Avoid re-downloading datasets

echo "Evaluation of $MODEL completed successfully!"

# lm_eval --model hf --model_args "pretrained=meta-llama/Llama-3.1-8B-Instruct,return_token_details=True" --tasks winogrande --device cuda:0 --batch_size 8 --output_path "results_scratch/test" --apply_chat_template
