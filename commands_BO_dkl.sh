# nohup bash commands_BO_vae.sh > logs/fullrun_vae_18aug_1.log 2>&1 &
# TASKS=("commonsense_qa" "gsm8k" "headqa_en")   # GPU 0

# TASKS=("pubmedqa" "sciq" "triviaqa")   # GPU 0        
# TASKS=("truthfulqa_gen" "wikitext" "mmlu" "ai2_arc") # GPU 4 

# 22 aug
# TASKS=("gsm8k" "headqa_en" "pubmedqa")   # GPU 2 (500s, started on Sat)
# TASKS=("commonsense_qa" "triviaqa") #GPU 0 (500s) DONE
# TASKS=("sciq" "truthfulqa_gen" "wikitext" "mmlu") # GPU 3 (500s, started on Sat)
# TASKS=("mmlu") #GPU4 (500s, started on 25 aug, speed)


TASKS=("gsm8k") # GPU 4 gsm8k_29aug
'''
5 Sep:
GPU 4: gsm8k, dim=8, hidden=128
GPU 5: gsm8k, dim=8, hidden=64
GPU 6: gsm8k, dim=8, hidden=32
GPU 7: gsm8k, dim=10, hidden=64

8 Sep:
GPU 7: gsm8k, dim=10, hidden=128
'''

# TASKS=("triviaqa")   # GPU 5
# TASKS=("commonsense_qa")   # GPU 6
# TASKS=("headqa_en")   # GPU 7

MODES=("dkl")
GPUS=(7)
# for task in "${TASKS[@]}"; do
#   mkdir -p "printout_BO/full_run_dkl/${task}"
#   for i in "${!MODES[@]}"; do
#     mode="${MODES[$i]}"
#     gpu="${GPUS[$i]}"
#     echo "Running task=$task, mode=$mode on GPU=$gpu"
#     CUDA_VISIBLE_DEVICES=$gpu python3 -u BO_runs_LLM_joint_optimization.py \
#       --contaminate=0 \
#       --iterations=62 \
#       --num_data=500 \
#       --epochs=1 \
#       --trials=3 \
#       --evaluation_cuda=0 \
#       --sample_method=random \
#       --eval_tasks="$task" \
#       --experiments_setting=ood \
#       --output_dir=results_full_run_dkl/ \
#       --lora_rank=128 \
#       --time_limit=500 \
#       --ucb_beta=5 \
#       --limit=100 \
#       --run_BO_on="$mode" \
#       --dkl_feature_dim=10 \
#       --dkl_hidden=128 \
#       > "printout_BO/full_run_dkl/${task}/${mode}.out" \
#       2> "printout_BO/full_run_dkl/${task}/${mode}.err" &
#     echo "Launched $task-$mode on GPU $gpu with PID=$!" >> pid_log.txt
#   done
#   wait
# done
# wait
set -Eeuo pipefail

USE_LOCAL_ONLY="${USE_LOCAL_ONLY:-0}"

for task in "${TASKS[@]}"; do
  LOG_DIR="printout_BO/full_run_dkl_bad/${task}_10dim128layer"
  mkdir -p "$LOG_DIR"

  for i in "${!MODES[@]}"; do
    mode="${MODES[$i]}"
    gpu="${GPUS[$i]}"

    RUN_ID="${task}_${mode}_gpu${gpu}_$$_$(date +%s)"

    export TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export HF_HUB_DISABLE_TELEMETRY=1
    export HF_HUB_DISABLE_PROGRESS_BARS=1

    export HF_HOME="$PWD/.hf_cache/$RUN_ID"
    export TRANSFORMERS_CACHE="$HF_HOME/transformers"
    export HF_DATASETS_CACHE="$HF_HOME/datasets"
    export TMPDIR="$PWD/.tmp/$RUN_ID"
    mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TMPDIR"

    OUT_DIR="results_full_run_dkl/${task}_10dim128layer/${mode}_${RUN_ID}"
    mkdir -p "$OUT_DIR"

    STDOUT_FILE="${LOG_DIR}/${mode}_${RUN_ID}.out"
    STDERR_FILE="${LOG_DIR}/${mode}_${RUN_ID}.err"

    echo "Running task=${task}, mode=${mode} on GPU=${gpu} (RUN_ID=${RUN_ID})"

    LOCAL_FLAG=""
    HF_TOKEN_SRC=""
    if [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
      export HF_TOKEN="${HUGGINGFACE_HUB_TOKEN}"
      HF_TOKEN_SRC="HUGGINGFACE_HUB_TOKEN"
    elif [[ -n "${HF_TOKEN:-}" ]]; then
      export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
      HF_TOKEN_SRC="HF_TOKEN"
    elif [[ -f "$HOME/.huggingface/token" ]]; then
      export HUGGINGFACE_HUB_TOKEN="$(cat "$HOME/.huggingface/token")"
      export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
      HF_TOKEN_SRC="~/.huggingface/token"
    elif [[ -f "$HOME/.cache/huggingface/token" ]]; then
      export HUGGINGFACE_HUB_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
      export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
      HF_TOKEN_SRC="~/.cache/huggingface/token"
    fi

    if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
      echo "ERROR: No Hugging Face token found. Set HUGGINGFACE_HUB_TOKEN or run 'huggingface-cli login'." >&2
      exit 1
    fi
    
    CUDA_VISIBLE_DEVICES="${gpu}" \
    python3 -u BO_runs_LLM_joint_optimization.py \
      --contaminate=0 \
      --iterations=62 \
      --num_data=500 \
      --epochs=1 \
      --trials=3 \
      --evaluation_cuda=0 \
      --sample_method=random \
      --eval_tasks="${task}" \
      --experiments_setting=ood \
      --output_dir="${OUT_DIR}" \
      --lora_rank=128 \
      --time_limit=500 \
      --ucb_beta=5 \
      --limit=100 \
      --run_BO_on="${mode}" \
      --seed=0 \
      --dkl_feature_dim=8 \
      --dkl_hidden=128 \
      ${LOCAL_FLAG} \
      > "${STDOUT_FILE}" \
      2> "${STDERR_FILE}" &

    echo "Launched ${task}-${mode} on GPU ${gpu} with PID=$! (RUN_ID=${RUN_ID})" >> pid_log.txt
  done

  wait
done

wait