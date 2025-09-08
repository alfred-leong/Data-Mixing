# run BO over only model parameters only (lora rank, layers to apply etc)

# Run BO on 5 datasets only. Experimenting to see if dimensionality is a problem.
# CUDA_VISIBLE_DEVICES=0 python3 -u BO_runs_LLM_joint_optimization.py --contaminate=0 --iterations=100 --num_data=5000 --epochs=1 --trials=1 --evaluation_cuda=0 --sample_method=random --eval_tasks=gsm8k --experiments_setting=ood --output_dir=output_joint_optimization/results_updated --lora_rank=128 --time_limit=100 --ucb_beta=0.5 --limit=1000 --run_BO_on=model >> printout_BO/BO_model_dims.out
CUDA_VISIBLE_DEVICES=1 python3 -u BO_runs_LLM_joint_optimization.py --contaminate=0 --iterations=100 --num_data=5000 --epochs=1 --trials=1 --evaluation_cuda=0 --sample_method=random --eval_tasks=gsm8k --experiments_setting=ood --output_dir=output_joint_optimization/results_updated --lora_rank=128 --time_limit=100 --ucb_beta=0.5 --limit=1000 --run_BO_on=all >> printout_BO/BO_all_dims_full.out
