
# run 1 epoch of data training for a huge range of LLM model params (sweep lora rank from lora_r_start to lora_r_end via lora_r_intervals, sweep through layers to apply 1 at a time)
# CUDA_VISIBLE_DEVICES=0 nohup python3 LORA_joint_opt_1_epoch_diff_model_size.py --domain=gsm8k --few_shot=1 --trials=3 --batch=2 --adjustable_module=q_proj,v_proj,k_proj --reverse_layers=0 --lora_r_start=8 --lora_r_end=20 --lora_r_intervals=8 > nohup_train_gsm8k_range_of_model_layers_reverse=0

# # run data training for different model variation with a specific config of which layer to apply e.g., [0,0,0,1,1] -> There are 5 different modules we can apply to. Q, V, K, Up, Down projection modules. This
# indicates we apply lora to Up and Down modules only.
# --train_time=1000000 and -epochs=1 means we train for 1 epoch with no time limit.
# we fix --lora_r=16 --num_layers=16
CUDA_VISIBLE_DEVICES=4 nohup python3 LORA_run_with_specific_config.py --train_time=1000000 --setting=ood --task=commonsense_qa --epochs=1 --total_data=5000 --lora_r=16 --num_layers=16 >> LORA_run_with_specific_config_commonsense_full_epoch.out
