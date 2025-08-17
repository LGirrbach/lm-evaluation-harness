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
  "EleutherAI/gpt-j-6B"
  "EleutherAI/gpt-neox-20b"
  "facebook/opt-6.7b"
  "facebook/opt-13b"
  "facebook/opt-1.3b"
  "tiiuae/falcon-7b"
  "tiiuae/falcon-7b-instruct"
  "tiiuae/falcon-40b-instruct" # >20B, optional
  "mosaicml/mpt-7b"
  "mosaicml/mpt-7b-instruct"
  "bigscience/bloom-7b1"
  "bigscience/bloomz-7b1"
  "meta-llama/Llama-2-7b-hf"
  "meta-llama/Llama-2-13b-hf"
  "meta-llama/Llama-2-7b-chat-hf"
  "meta-llama/Llama-2-13b-chat-hf"
  "allenai/tulu-7b"
  "allenai/tulu-13b"
  "OpenAssistant/oasst-sft-6-llama-30b" # optional large
  "cerebras/Cerebras-GPT-6.7B"
)

# Popular multiple-choice tasks
TASKS="arc_easy,arc_challenge,hellaswag,winogrande,boolq,piqa,openbookqa,commonsense_qa,headqa,prost"

# Number of few-shot examples
FEWSHOT=5

# Get the model for this array task
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
SAFE_MODEL_NAME=$(echo "$MODEL" | tr '/:' '_')
OUT_FILE="$RESULTS_DIR/${SAFE_MODEL_NAME}.json"
LOG_FILE="$RESULTS_DIR/${SAFE_MODEL_NAME}_samples.jsonl"

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
