#!/bin/bash
#SBATCH --job-name=model-evaluation           # Name of the job
#SBATCH --nodes=1                            # Request 1 nodes
#SBATCH --ntasks-per-node=1                  # Start only 1 task per node
#SBATCH --gres=gpu:1                         # Request 1 GPU
#SBATCH --cpus-per-task=8                    # Request 8 cpus per task
#SBATCH --mem=70GB                           # Request 70GB of cpu memory per node
#SBATCH --time=12:00:00                      # Time limit hrs:min:sec (adjust as needed)
#SBATCH --output=slurm_logs/evaluation/output/evaluate_models_%a.out  # Output file (%A=array job ID, %a=array task ID)
#SBATCH --error=slurm_logs/evaluation/error/evaluate_models_%a.err    # Error log file
#SBATCH --partition=mcml-hgx-h100-94x4,mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --array=0-15

source /dss/dsshome1/04/go25kod3/miniforge3/etc/profile.d/conda.sh
conda activate lm-eval-harness

set -euo pipefail

# Directory to store results
RESULTS_DIR="./results_scratch"

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
  # Other Models
  "openai/gpt-oss-20b"
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
    --apply_chat_template \
    --device cuda:0

echo "Evaluation of $MODEL completed successfully!"

# lm_eval --model hf --model_args "pretrained=Qwen/Qwen3-14B,return_token_details=True,trust_remote_code=True" --tasks winogrande --device cuda:0 --batch_size 8 --output_path "results_scratch/test" --apply_chat_template
