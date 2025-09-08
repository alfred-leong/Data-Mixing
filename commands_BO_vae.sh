# VAE-BO Dim 10
# CUDA_VISIBLE_DEVICES=3 python3 -u BO_runs_LLM_joint_optimization.py \
#   --contaminate=0 \
#   --iterations=100 \
#   --num_data=5000 \
#   --epochs=1 \
#   --trials=3 \
#   --evaluation_cuda=0 \
#   --sample_method=random \
#   --eval_tasks=headqa_en \
#   --experiments_setting=ood \
#   --output_dir=output_joint_optimization/results_updated \
#   --lora_rank=128 \
#   --time_limit=100 \
#   --ucb_beta=5 \
#   --limit=100 \
#   --run_BO_on=vae \
#   --vae_dim=10 \
#   --vae_hidden=64 \
#   --vae_epochs=20 \
#   --vae_lr=1e-3 \
#   > results/headqa_en/BO_joint_vae_dim10_13aug.out

# # VAE-BO Dim 8
  # CUDA_VISIBLE_DEVICES=4 python3 -u BO_runs_LLM_joint_optimization.py \
  # --contaminate=0 \
  # --iterations=100 \
  # --num_data=5000 \
  # --epochs=1 \
  # --trials=5 \
  # --evaluation_cuda=0 \
  # --sample_method=random \
  # --eval_tasks=headqa_en \
  # --experiments_setting=ood \
  # --output_dir=output_joint_optimization/results_updated \
  # --lora_rank=128 \
  # --time_limit=100 \
  # --ucb_beta=0.5 \
  # --limit=100 \
  # --run_BO_on=vae \
  # --vae_dim=8 \
  # --vae_hidden=64 \
  # --vae_epochs=20 \
  # --vae_lr=1e-3 \
  # > results/headqa_en/BO_joint_vae_dim8.out

# nohup bash commands_BO_vae.sh > logs/fullrun_vae_18aug_1.log 2>&1 &
# TASKS=("commonsense_qa" "gsm8k" "headqa_en")   # GPU 0

# TASKS=("pubmedqa" "sciq" "triviaqa")   # GPU 0        
# TASKS=("truthfulqa_gen" "wikitext" "mmlu" "ai2_arc") # GPU 4 

# 22 aug
TASKS=("gsm8k" "commonsense_qa" "triviaqa")   # GPU 2 (500s, started on Sat)
# TASKS=("commonsense_qa" "triviaqa") #GPU 0 (500s) DONE
# TASKS=("sciq" "truthfulqa_gen" "wikitext" "mmlu") # GPU 3 (500s, started on Sat)
# TASKS=("mmlu") #GPU4 (500s, started on 25 aug, speed)

#Reruns


MODES=("vae")
GPUS=(3)
for task in "${TASKS[@]}"; do
  mkdir -p "printout_BO/full_run_vae_500samples/${task}"
  for i in "${!MODES[@]}"; do
    mode="${MODES[$i]}"
    gpu="${GPUS[$i]}"
    echo "Running task=$task, mode=$mode on GPU=$gpu"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u BO_runs_LLM_joint_optimization.py \
      --contaminate=0 \
      --iterations=62 \
      --num_data=500 \
      --epochs=1 \
      --trials=3 \
      --evaluation_cuda=0 \
      --sample_method=random \
      --eval_tasks="$task" \
      --experiments_setting=ood \
      --output_dir=results_full_run_vae_500samples/ \
      --lora_rank=128 \
      --time_limit=500 \
      --ucb_beta=5 \
      --limit=100 \
      --run_BO_on="$mode" \
      --vae_dim=10 \
      --vae_lr=5e-4 \
      --vae_hidden=128 \
      > "printout_BO/full_run_vae_500samples/${task}/${mode}.out" \
      2> "printout_BO/full_run_vae_500samples/${task}/${mode}.err" &
    echo "Launched $task-$mode on GPU $gpu with PID=$!" >> pid_log.txt
  done
  wait
done
wait