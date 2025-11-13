from botorch.models import SingleTaskGP
from botorch.acquisition import UpperConfidenceBound, LogExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from botorch.models.transforms.outcome import Standardize
import shutil
import torch
from itertools import product
import numpy as np
import random
from typing import Optional, List
import gc
import time

from helper import get_data_from_mixing_ratio
from image_training import train
from typing import List
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training ,
    set_peft_model_state_dict,
)

lora_alpha = 16
lora_dropout= 0.05
lora_r=16
lora_target_modules = [
    "q_proj",
    "v_proj",
]
lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=lora_target_modules,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)

# Convert all elements to float or int for JSON serialization
def to_serializable(val):
    if isinstance(val, torch.Tensor):
        val = val.item()
    return float(val) if isinstance(val, float) or isinstance(val, np.floating) else int(val) if isinstance(val, int) or isinstance(val, np.integer) else val
    
def iterative_loop(data_sources : List[DataLoader], validation_data : DataLoader, method : str, additional_info : List[List[float]], seed, layers_freeze : int, cuda : str, num_epochs=10, iterations=10, data="images", printout=True):
    
    input_X = torch.Tensor((len(data_sources))*[float(1/len(data_sources))]) # initial X
    GP_input = []
    observed_output = []
    for i in range(iterations):
        print("iteration: ", i)
        
        if printout:
            print("mixing data with method: ", method)

        mixed_data = get_data_from_mixing_ratio(data_sources, mixing_ratio=input_X,
                                                additional_info=additional_info,
                                                method=method,
                                                seed=seed,
                                                base_number_of_batches=20) # each agent do some influence function process to get data
        
        if data=="images":
            acc_all, observed_performance, _ = train(mixed_data, validation_data, seed=seed, lr=5e-5, cuda=cuda, num_epochs=num_epochs, num_layer_to_unfreeze=layers_freeze, printout=printout) # observe the performance of this dataset from finetuning
        if printout:
            print("performance after training: ", observed_performance)
        # format the observed performance and current parameters for this round with previously seen values
        current_gp_input = list(input_X)
        #current_gp_input.append(current_mixing_parameter)
        GP_input.append(current_gp_input)
        observed_output.append(observed_performance)
        
        # fit the GP with previous selected parameters and observed performance from this round
        
        gp = SingleTaskGP(torch.DoubleTensor(GP_input), torch.DoubleTensor(observed_output).reshape(-1,1), outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        
        # use Bayesian Optimization to propose next candidate mixing parameter and score parameters for agents
        UCB = UpperConfidenceBound(gp, beta=1)
        bounds = torch.stack([torch.zeros(len(current_gp_input)), torch.ones(len(current_gp_input))]) # need to change the bounds for parameters
        A = [1.0] * len(data_sources)
        x = list(range(len(data_sources)))
        candidate, acq_value = optimize_acqf(
            UCB, bounds=bounds, q=1, num_restarts=20, raw_samples=50,
            #equality_constraints = [(torch.tensor(list(range(len(data_sources)))), torch.tensor([1.0] * len(data_sources)), 1)]
            equality_constraints = [(torch.tensor(x), torch.tensor(A), 1)]
        )
        input_X = [x if x >= 0.05 else 0 for x in candidate[0]]
        if printout:
            print("proposed parameters for next round by BO:", input_X)
    return GP_input, observed_output, gp

def get_BO_plots(observations):
    BO_to_plot = []
    for x in range(0,len(observations)):
        BO_to_plot.append((max(observations[:x+1])))
    return BO_to_plot

def run_BO(all_loaders, validaton_dataloader, method, additional_info, seed, iterations, num_epochs, cuda, layers_freeze, printout=False):
    print("running BO...")
    X, observations, gp = iterative_loop(all_loaders, validaton_dataloader, cuda=cuda, method=method, additional_info=additional_info, layers_freeze=layers_freeze, seed=seed, num_epochs=num_epochs, iterations=iterations, printout=printout)
    BO_to_plot = get_BO_plots(observations) # BO results
    naive_combine = BO_to_plot[0] # naive mixing result is the first iteration result of BO

    def get_optimal_mixture_from_GP_posterior():
        UCB = UpperConfidenceBound(gp, beta=0.0)
        bounds = torch.stack([torch.zeros(len(all_loaders)), torch.ones(len(all_loaders))]) # need to change the bounds for parameters
        A = [1.0] * len(all_loaders)
        x = list(range(len(all_loaders)))
        candidate, acq_value = optimize_acqf(
            UCB, bounds=bounds, q=1, num_restarts=20, raw_samples=30,
            equality_constraints = [(torch.tensor(x), torch.tensor(A), 1)]
        )
        return candidate

    def get_best_observation_mixture():
        
        # Find the index in list B that has the highest value
        highest_index = observations.index(max(observations))
        
        # Return the corresponding item in list A
        return X[highest_index]

    
    print("best mixture found in BO iterations is: ", get_best_observation_mixture())
    return BO_to_plot

from LLM.llm import load_data, get_tokenizer_and_model, extract_data_mixture_and_train
from LLM.llm import extract_data_mixture_and_train, evaluate_tasks, load_data, get_tokenizer_and_model

from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback
from tqdm import tqdm
import os
import json
import lm_eval

def run_BO_for_LLM(data_domains : List[str], random_dir : str, BO_run : int, total_data : int, evaluation_cuda : str, evaluation_task : dict, sampling_method = "random", train_epochs : int = 1, training_batch : int = 8, evaluation_batch : int = 4, printout=True, max_steps = -1, eval_steps=100, model_id = "LLM/llama_8b_instruct"):
    train_datasets = []
    val_datasets = []
    for data_domain in data_domains:
        train_dataset, val_dataset = load_data(data_domain=data_domain)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    # get tokenizer and model
    tokenizer, model = get_tokenizer_and_model(model_id = model_id)
    #model = prepare_model_for_kbit_training(model)

    input_X = (len(data_domains))*[float(1/len(data_domains))] # initial X is balanced all
    GP_input = []
    observed_output = []

    all_influences = []
    for train_domain in data_domains:
        all_influences.append(None)
        #all_influences.append(torch.load("influence/"+str(train_domain)+"_training.pt"))
    
    for i in tqdm(range(BO_run)):
        print("iteration: ", i)
        
        if printout:
            print("mixing data with method: ", sampling_method)

        # sample from each domain and train a model
        path_to_final_model = extract_data_mixture_and_train(model=model, random_dir=random_dir, tokenizer=tokenizer, 
                                                        train_datasets=train_datasets, 
                                                        val_datasets=val_datasets, 
                                                        data_domains=data_domains, 
                                                        mixing_ratio=input_X, 
                                                        additional_info=all_influences, # add IF value
                                                        total_number_datapoints=total_data, 
                                                        run_name="BO_run_" +str(i),
                                                        method=sampling_method,
                                                        train_epochs=train_epochs, 
                                                        batch_size=training_batch,
                                                        max_step=max_steps,
                                                        lora_config=lora_config,
                                                        eval_steps=eval_steps)
        # free gpu memory
        with torch.no_grad():
            torch.cuda.empty_cache()
        print("evaluating...")
        lora_path = path_to_final_model #final_model_after_training
        config = PeftConfig.from_pretrained(lora_path)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype='auto')
        lora_model = PeftModel.from_pretrained(model, lora_path).to(evaluation_cuda)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True,)
        
        observed_performance = 0
        tasks = list(evaluation_task.keys())
        lora_model.eval()
        results=evaluate_tasks(tasks, lora_model, tokenizer, evaluation_batch)
        print("deleting lora model after evaluation.")
        shutil.rmtree(lora_path, ignore_errors=True)
        print("results: ", results["results"])
        for task in evaluation_task:
            task_weight, metric = evaluation_task[task]
            print(task_weight)
            print(metric)
            print(results["results"][task][metric])
            perf = results["results"][task][metric]
            if task == "wikitext":
                perf = - perf # we want to maximize the score, so for perplexity we maximize instead
            observed_performance += (perf * task_weight)
        print("current iteration performance: ", observed_performance)
        lora_model.to("cpu")
        # format the observed performance and current parameters for this round with previously seen values
        current_gp_input = list(input_X)
        
        #current_gp_input.append(current_mixing_parameter)
        GP_input.append(current_gp_input)
        observed_output.append(observed_performance)
        
        # fit the GP with previous selected parameters and observed performance from this round
        
        gp = SingleTaskGP(torch.DoubleTensor(GP_input), torch.DoubleTensor(observed_output).reshape(-1,1), outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        
        # use Bayesian Optimization to propose next candidate mixing parameter and score parameters for agents
        UCB = UpperConfidenceBound(gp, beta=1)
        bounds = torch.stack([torch.zeros(len(current_gp_input)), torch.ones(len(current_gp_input))]) # need to change the bounds for parameters
        A = [1.0] * len(data_domains)
        x = list(range(len(data_domains)))
        candidate, acq_value = optimize_acqf(
            UCB, bounds=bounds, q=1, num_restarts=20, raw_samples=50,
            #equality_constraints = [(torch.tensor(list(range(len(data_sources)))), torch.tensor([1.0] * len(data_sources)), 1)]
            equality_constraints = [(torch.tensor(x), torch.tensor(A), 1)]
        )
        input_X = [x if x >= 0.05 else 0 for x in candidate[0]]
        if printout:
            print("proposed parameters for next round by BO:", input_X)
    return GP_input, observed_output, gp

def arrange_lora_config(lora_r, lora_dropout, num_layers_to_apply, five_dim_vector, alpha=16):
    '''
    lora_r: float
    lora_dropout = float 
    num_layers_to_apply = int
    five_dim_vector = List[float]. Five dimension
    '''
    lora_r = int(lora_r)
    num_layers_to_apply = int(num_layers_to_apply)
    alpha = int(alpha)
    five_dim_vector = [int(x) for x in five_dim_vector] # convert to int
    print("arranging lora config with parameters: ", lora_r, lora_dropout, alpha, num_layers_to_apply, five_dim_vector)

    # only .mlp layers have up, down, gate proj
    # only .self_attn layers have q, v, k proj
    # ["model.layers.0.self_attn.k_proj"]
    lora_modules_all = ["q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"]
    lora_module_to_tune = [mod for mod, flag in zip(lora_modules_all, five_dim_vector) if flag == 1]
    lora_specific_modules = []
    print(lora_module_to_tune)
    for module in lora_module_to_tune:
        if module == "q_proj" or module == "v_proj" or module == "k_proj":
            for i in range(num_layers_to_apply):
                lora_specific_modules.append("model.layers."+str(i)+".self_attn."+module)
        else:
            for i in range(num_layers_to_apply):
                lora_specific_modules.append("model.layers."+str(i)+".mlp."+module)
    
    # if we choose all 0
    if len(lora_specific_modules) == 0:
        return None

    # lora r is chosen as 0
    if lora_r == 0:
        return None
    config = LoraConfig(
    r=lora_r,
    lora_alpha=alpha,
    target_modules=lora_specific_modules,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",)

    return config
    '''
    model.layers.0.self_attn
    model.layers.0.self_attn.q_proj
    model.layers.0.self_attn.k_proj
    model.layers.0.self_attn.v_proj
    model.layers.0.self_attn.o_proj
    model.layers.0.self_attn.rotary_emb
    model.layers.0.mlp
    model.layers.0.mlp.gate_proj
    model.layers.0.mlp.up_proj
    model.layers.0.mlp.down_proj
    model.layers.0.mlp.act_fn
    model.layers.0.input_layernorm
    model.layers.0.post_attention_layernorm
    '''

def joint_opt_BO_LLM_only_data(default_rank, default_layer, default_num_layers_to_apply, default_dropout, default_alpha, time_callback, data_domains : List[str], random_dir : str, BO_run : int, total_data : int, evaluation_cuda : str, evaluation_task : dict, ucb_beta, trial_number, sampling_method = "random", train_epochs : int = 1, training_batch : int = 8, evaluation_batch : int = 4, printout=True, max_steps = -1, eval_steps=100, limit=100, model_id = "LLM/llama_8b_instruct", seed = 42, output_dir= "results/"):
    
    train_datasets = []
    val_datasets = []
    for data_domain in data_domains:
        train_dataset, val_dataset = load_data(data_domain=data_domain)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
    
    # mixing ratio
    input_X = (len(data_domains))*[float(1/len(data_domains))] # initial X is balanced all
    input_X_between_0_1 = (len(data_domains))*[float(1/len(data_domains))]

    # mixing ratio bounds
    lower_bound = [0.0] * (len(data_domains))
    upper_bound = [1.0] * (len(data_domains))
    
    bounds = torch.stack([torch.tensor(lower_bound), torch.tensor(upper_bound)])
    
    GP_input = []
    observed_output = []

    all_influences = []
    for train_domain in data_domains:
        all_influences.append(None)
        #all_influences.append(torch.load("influence/"+str(train_domain)+"_training.pt"))
    
    results_list = []
    max_performance_so_far = float('-inf')
    dataset = "_".join(evaluation_task.keys())
    run_BO_on = "data"
    results_dir = f"{output_dir}/{dataset}/{run_BO_on}"
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"trial_{trial_number + 1}.json")
    meta_info = {
        "seed": seed,
        "initial_model_params": [default_num_layers_to_apply, default_layer, default_rank, default_dropout]
    }
    meta_path = os.path.join(results_dir, f"trial_{trial_number + 1}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta_info, f, indent=2)

    for i in tqdm(range(BO_run)):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        tokenizer, model = get_tokenizer_and_model(model_id = model_id)

        print("iteration: ", i)
        print("input_X: ", input_X) 
        
        lora_config = arrange_lora_config(default_rank, default_dropout, default_num_layers_to_apply, default_layer)
        
        if lora_config is None:
            observed_performance = -1e6 # very bad performance if we use this
        else: # sample from each domain and train a model
            path_to_final_model = extract_data_mixture_and_train(model=model, random_dir=random_dir, tokenizer=tokenizer, 
                                                            train_datasets=train_datasets, 
                                                            val_datasets=val_datasets, 
                                                            data_domains=data_domains, 
                                                            mixing_ratio=input_X[:len(data_domains)], 
                                                            additional_info=all_influences, # add IF value
                                                            total_number_datapoints=total_data, 
                                                            run_name="BO_run_" +str(i),
                                                            method=sampling_method,
                                                            train_epochs=train_epochs, 
                                                            batch_size=training_batch,
                                                            max_step=max_steps,
                                                            lora_config=lora_config,
                                                            eval_steps=eval_steps, callback=[time_callback], seed=seed)
            # free gpu memory
            with torch.no_grad():
                torch.cuda.empty_cache()
            print("evaluating...")
            lora_path = path_to_final_model
            config = PeftConfig.from_pretrained(lora_path)
            model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype='auto')
            lora_model = PeftModel.from_pretrained(model, lora_path).to(evaluation_cuda)
            tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True,)
            
            observed_performance = 0
            tasks = list(evaluation_task.keys())
            results=evaluate_tasks(tasks, lora_model, tokenizer, evaluation_batch,few_shot=1, limit=limit)
            print("deleting lora model after evaluation.")
            base_path = lora_path.rsplit('/', 1)[0] + '/'
            shutil.rmtree(base_path, ignore_errors=True)
            for task in evaluation_task:
                task_weight, metric = evaluation_task[task]
                if task == "ai2_arc":
                    perf = results["results"]["arc_challenge"][metric]
                else:
                    perf = results["results"][task][metric]
                if task == "wikitext":
                    perf = - perf # we want to maximize the score, so for perplexity we maximize instead
                observed_performance += (perf * task_weight)
            lora_model.to("cpu")
        print("current iteration weighted performance: ", observed_performance)
        # format the observed performance and current parameters for this round with previously seen values
        current_gp_input = list(input_X_between_0_1)
        
        GP_input.append(current_gp_input)
        observed_output.append(observed_performance)
        
        # Save results for this iteration
        max_performance_so_far = max(max_performance_so_far, observed_performance)
        results_list.append({
            "iteration": i + 1,
            "mixing_ratio": [to_serializable(x) for x in input_X],
            "performance": observed_performance,
            "max_performance_so_far": max_performance_so_far
        })
        # Write to JSON after each iteration for safety
        with open(results_path, "w") as f:
            json.dump(results_list, f, indent=2)

        # fit the GP with previous selected parameters and observed performance from this round
        gp = SingleTaskGP(torch.DoubleTensor(GP_input), torch.DoubleTensor(observed_output).reshape(-1,1), outcome_transform=Standardize(m=1))
        print("GP past observed values (should be between [0,1]): ", GP_input)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        
        # use Bayesian Optimization to propose next candidate mixing parameter and score parameters for agents
        UCB = UpperConfidenceBound(gp, beta=ucb_beta)
        #logNEI = LogExpectedImprovement(model=gp, best_f=max(observed_output))
        A = [1.0] * len(data_domains)
        x = list(range(len(data_domains)))
        candidate, acq_value = optimize_acqf(
            UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=10,
            equality_constraints = [(torch.tensor(x), torch.tensor(A), 1)] # edit this TODO.
        )
        def process_values(values, data_domains_len):
            result = []
            
            # Step 1: Squash first `data_domains_len` elements if less than 0.05
            for v in values[:data_domains_len]:
                result.append(0 if v.item() < 0.05 else v)

            # Step 1.5: Normalize the first `data_domains_len` elements to sum to 1
            sum_values = sum(result)
            if sum_values > 0:
                result = [v / sum_values for v in result]
                
            print("proposed candidate after processing:", result)
            return result
        
        print("proposed candidate before processing:", candidate[0])
        # these are updated with the candidates and used in next iteration
        input_X_between_0_1 = list(candidate[0])
        input_X = process_values(candidate[0], len(data_domains))
        
    return GP_input, observed_output, gp

class TimeBasedLMEvalCallback(TrainerCallback):
    """
    A custom callback to run lm_eval.simple_evaluate at set wall-clock time intervals 
    (e.g., every 20 seconds).
    """
    def __init__(self, model, tokenizer, eval_interval_seconds, eval_tasks, eval_limit, results_dir):
        self.tokenizer = tokenizer
        self.eval_interval_seconds = eval_interval_seconds
        self.eval_tasks = eval_tasks
        self.eval_limit = eval_limit
        self.results_dir = results_dir
        self.eval_history = []
        self.file_count = 1
        
        # Internal state to track time and steps
        self.start_time = None
        self.last_eval_time = 0.0
        self.cumulative_training_time = 0.0
        
        # Initialize TaskManager outside the loop for efficiency
        self.task_manager = lm_eval.tasks.TaskManager()

    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the start of training to record the initial time."""
        self.start_time = time.time()
        # Initialize a new variable to track time spent *training* since the last evaluation
        self.training_time_since_last_eval = 0.0 
        self.last_step_time = time.time() # To measure the time of the last step
        
        self._run_evaluation(state.global_step, kwargs['model'])  # Initial evaluation at step 0
        print(f"[{time.strftime('%H:%M:%S')}] Initial evaluation done. Start training... Wall-clock evaluation will run every {self.eval_interval_seconds} seconds of actual training.")
        

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of every training step."""
        
        # 1. Calculate the time taken for the last training step
        current_step_time = time.time()
        time_for_this_step = current_step_time - self.last_step_time
        
        # 2. Add this time to the counter for training time
        self.training_time_since_last_eval += time_for_this_step
        # NEW: Add to the cumulative total
        self.cumulative_training_time += time_for_this_step
        
        # 3. Update the last step time for the next iteration
        self.last_step_time = current_step_time 

        # 4. Check if the required interval of *training* time has passed
        if self.training_time_since_last_eval >= self.eval_interval_seconds:
            
            total_elapsed_time = time.time() - self.start_time
            print(f"\n[{time.strftime('%H:%M:%S')}] Wall-clock interval reached ({total_elapsed_time:.1f}s total elapsed). Running evaluation...")
            
            current_model = kwargs['model']
            # Run the lm_eval evaluation
            self._run_evaluation(state.global_step, current_model)
            
            # RESET the training time tracker, NOT the wall-clock time!
            self.training_time_since_last_eval = 0.0 
            
            control.should_log = True
            control.should_save = True
            
            print(f"\n[{time.strftime('%H:%M:%S')}] Resuming training.")

        control.should_log = False
        return control
    
    def _run_evaluation(self, global_step, current_model):
        """Helper function to execute the evaluation harness."""
        
        # IMPORTANT: Wrap the model for the evaluation harness
        from lm_eval.models.huggingface import HFLM
        
        # We assume the model is a HuggingFace model wrapper, and we pass the
        # actual, fine-tuned model's device/dtype information via the wrapper.
        lm_model = HFLM(
            pretrained=current_model,
            tokenizer=self.tokenizer,
            device=current_model.device.type,
        )
        
        # Note: We hardcode a small batch size to prioritize stability 
        # during the evaluation interrupt. Adjust as needed.
        results = lm_eval.simple_evaluate(
            model=lm_model,
            limit=self.eval_limit,
            tasks=list(self.eval_tasks.keys()),
            task_manager=self.task_manager,
            batch_size=4, 
            num_fewshot=0, 
            log_samples=False 
        )
        
        current_elapsed_time = time.time() - self.start_time
        step_results = {
            "global_step": global_step,
            "wall_clock_time_sec": current_elapsed_time,
            "cumulative_training_time_sec": self.cumulative_training_time
        }
        for task, value in self.eval_tasks.items():
            _, metric = value
            perf = results["results"][task][metric]

            step_results[f"{task}_{metric}"] = perf
            
            print(f"Eval Summary (Step {global_step} @ {current_elapsed_time:.1f}s):")
            print(f"  {metric}: {perf:.4f}")
        self.eval_history.append(step_results)
        self.last_step_time = time.time()

    def on_train_end(self, args, state, control, **kwargs):
        """Called when training ends to save the final results."""
        print(f"\nSaving evaluation history to {self.results_dir}...")
        output_file = os.path.join(self.results_dir, f"lm_eval_history_{self.file_count}.json")
        try:
            with open(output_file, 'w') as f:
                json.dump(self.eval_history, f, indent=4)
            print("Evaluation history successfully saved.")
            self.file_count += 1
            self.cumulative_training_time = 0.0
            self.eval_history = []
        except Exception as e:
            print(f"Error saving evaluation history: {e}")

class JsonLogCallback(TrainerCallback):
    """
    A custom callback to save all log entries (loss, metrics) to a JSON file.
    """
    def __init__(self, output_dir: str):
        # Create a unique filename for this run's logs
        self.count = 0
        self.output_dir = output_dir
        self.log_data = []

    def on_log(self, args: TrainingArguments, state, control, logs=None, **kwargs):
        """
        Intercepts log events (which occur during training loss logging and evaluation).
        """
        if logs is not None:
            # Add the global step and epoch for context
            logs['step'] = state.global_step
            logs['epoch'] = state.epoch
            
            self.log_data.append(logs)

    def on_train_end(self, args: TrainingArguments, state, control, **kwargs):
        """
        Ensures all final logs are written when training finishes.
        """
        self._write_logs()

    def _write_logs(self):
        """
        Helper method to write the collected log data to the JSON file.
        """
        try:
            log_filename = f"run_log_{self.count}.json"
            log_path = os.path.join(self.output_dir, log_filename)
            with open(log_path, 'w') as f:
                # Use json.dump to write the list of dictionaries
                json.dump(self.log_data, f, indent=4)
            self.count += 1
            self.log_data = []
        except Exception as e:
            # Handle potential file writing errors
            print(f"Error writing to log file {log_path}: {e}")

class ElapsedTimeCallback(TrainerCallback):
    """A custom callback to log the elapsed time since the start of training at each evaluation."""
    
    def on_train_begin(self, args: TrainingArguments, state, control, **kwargs):
        # Record the start time of the entire training process
        self.start_time = time.time()
        print("Training started. Start time recorded.") # For confirmation

    def on_evaluate(self, args: TrainingArguments, state, control, metrics=None, **kwargs):
        
        # Calculate elapsed time
        elapsed_seconds = time.time() - self.start_time

        # 1. Inject the elapsed time into the metrics dictionary
        if metrics is not None:
            metrics['eval_elapsed_time_s'] = round(elapsed_seconds, 2)
        
        print("metrics: ", metrics)
        # 2. Retrieve the trainer object from kwargs (it's always passed here)
        trainer = kwargs.get('trainer')

        if trainer and metrics:
            trainer.log(metrics) 
            control.should_log = True
            
        return control # Always return the control object

def joint_opt_BO_LLM(time_callback, lora_rank_max, data_domains : List[str], random_dir : str, BO_run : int, total_data : int, evaluation_cuda : str, evaluation_task : dict, ucb_beta, trial_number, sampling_method = "random", train_epochs : int = 1, training_batch : int = 8, evaluation_batch : int = 4, printout=True, max_steps = -1, eval_steps=100, limit=100, model_id = "LLM/llama_8b_instruct", seed = 42, output_dir = "results/"):

    train_datasets = []
    val_datasets = []
    for data_domain in data_domains:
        train_dataset, val_dataset = load_data(data_domain=data_domain)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    # get tokenizer and model
    model_id = "Qwen/Qwen3-14B"
    tokenizer, model = get_tokenizer_and_model(model_id = model_id)
    lora_max_num_layers = len(model.model.layers)
    
    # for discrete BO; not used here.
    fixed_features_list =[{len(data_domains)+2:0},{len(data_domains)+2:1},
                          {len(data_domains)+3:0},{len(data_domains)+3:1},
                          {len(data_domains)+4:0},{len(data_domains)+4:1},
                          {len(data_domains)+5:0},{len(data_domains)+5:1},
                          {len(data_domains)+6:0},{len(data_domains)+6:1}]
    
    # input_X is the input to our GP:
    # first len(data_domains) are the mixing ratio for data (0 to 1, constrained to sum to 1)
    # next 1 dimension is the number of layers to apply to (integer)
    # next 5 dimension vector to indicate which layer to apply to (0 or 1)
    # then lora rank (integer)
    # then lora dropout (float)
    
    # input_X_between_0_1 is the standardized form of input_X (with everything between 0 and 1)
    # We use this input for the BO to make optimization more stable.
    
    # The following represents the inputs used for first iteration, so it's hard coded.
    # mixing ratio - even ratio for all domains for first iteration
    input_X = (len(data_domains))*[float(1/len(data_domains))] # initial X is balanced all
    input_X_between_0_1 = (len(data_domains))*[float(1/len(data_domains))]

    # Use bad mixing ratio
    # def get_mixing_ratio(evaluation_task):
    #     dataset = "_".join(evaluation_task.keys())
    #     if dataset == "gsm8k":
    #         return [0,0,0.14,0.31,0.12,0.14,0,0.29,0,0]
    #     elif dataset == "commonsense_qa":
    #         return [0,0,0,1,0,0,0,0,0,0]
    #     elif dataset == "headqa_en":
    #         return [0.1221754401922226,0.0,0.539222776889801,0.0,0.0,0.0,0.2574373185634613,0.0,0.0,0.0811644196510315]
    #     elif dataset == "pubmedqa":
    #         return [0.0,0.0,0.0,0.0,0.0,0.06087161973118782,0.9391283392906189,0.0,0.0,0.0]
    #     elif dataset == "triviaqa":
    #         return [0.0,0.0,0.0,0.6801438927650452,0.11240127682685852,0.0,0.2074548453092575,0.0,0.0,0.0]
    #     elif dataset == "truthfulqa_gen":
    #         return [0.0,0.0,0.0,0.0,0.0,0.20507599413394928,0.0,0.0,0.0,0.7949240207672119]
    #     else:   # For wikitext, mmlu, ai2_arc: no bad mixing ratio found, so we use arbitrary bad mixing ratio
    #         return [1,0,0,0,0,0,0,0,0,0]

    # input_X = get_mixing_ratio(evaluation_task)
    # input_X_between_0_1 = get_mixing_ratio(evaluation_task)
    
    # lora number of layers - use half the layers for first iteration
    input_X.append(int(lora_max_num_layers*0.5))
    input_X_between_0_1.append(0.5)
    
    # apply lora to all modules for first iteration
    input_X = input_X + [1, 1, 1, 1, 1] # 5 dimension vector to indicate apply to all layers as initial input
    input_X_between_0_1 = input_X_between_0_1 + [1, 1, 1, 1, 1]
    
    # lora rank of 72 for first iteration
    input_X.append(72)
    input_X_between_0_1.append(72.0/lora_rank_max)
    
    # lora dropout of 0.05 for first iteration
    input_X.append(0.05)
    input_X_between_0_1.append(0.05)

    # lora alpha of 72 for first iteration - Set equal to lora rank, with double lora_rank_max as max
    input_X.append(72)
    input_X_between_0_1.append(72.0/lora_rank_max*2)
    
    # next, define bounds for BO (which interval should our values lie)
    # Recall that BO operates with input_X_between_0_1, which squashed everything to be in [0,1]
    # mixing ratio bounds
    lower_bound = [0] * (len(data_domains))
    upper_bound = [1] * (len(data_domains))
    
    # lora number of layers bounds
    lower_bound.append(0)
    upper_bound.append(1)
    
    # which layer to apply to bounds
    lower_bound+=[0, 0, 0, 0, 0]
    upper_bound+=[1, 1, 1, 1, 1]
    
    # lora rank bounds
    lower_bound.append(0)
    upper_bound.append(1)
    
    # lora dropout bounds; this one is not in [0,1] but in [0,0.1]
    lower_bound.append(0.0)
    upper_bound.append(0.1)

    # lora alpha bounds
    lower_bound.append(0)
    upper_bound.append(1)
    
    # the actual bounds
    bounds = torch.stack([torch.tensor(lower_bound), torch.tensor(upper_bound)])
    
    GP_input = [] # X
    observed_output = [] # y

    all_influences = [] # not used currently
    for train_domain in data_domains:
        all_influences.append(None)
        #all_influences.append(torch.load("influence/"+str(train_domain)+"_training.pt"))
    
    results_list = []
    max_performance_so_far = float('-inf')
    dataset = "_".join(evaluation_task.keys())
    run_BO_on = "all"
    results_dir = f"{output_dir}/{dataset}/{run_BO_on}"
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"trial_{trial_number + 1}.json")
    meta_info = {
        "seed": seed,
    }
    meta_path = os.path.join(results_dir, f"trial_{trial_number + 1}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta_info, f, indent=2)

    # eval_callback = TimeBasedLMEvalCallback(
    #     model=model, 
    #     tokenizer=tokenizer,
    #     eval_interval_seconds=20,
    #     eval_tasks=evaluation_task,
    #     eval_limit=20,  # Number of eval samples
    #     results_dir=results_dir,
    # )
    elapsed_callback = ElapsedTimeCallback()
    json_logger_callback = JsonLogCallback(output_dir=output_dir)

    for i in tqdm(range(BO_run)):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model_id = "Qwen/Qwen3-14B"
        tokenizer, model = get_tokenizer_and_model(model_id = model_id)

        print("iteration: ", i)
        print("input_X: ", input_X)
        if printout:
            print("mixing data with method: ", sampling_method)
        
        # take the model related inputs and arrange them in a nice lora config file
        lora_config = arrange_lora_config(input_X[-3], input_X[-2], input_X[len(data_domains)], input_X[len(data_domains)+1:len(data_domains)+6], input_X[-1],)
        
        if lora_config is None:
            observed_performance = -1e6 # very bad performance if we use this
        else: # sample from each domain and train a model
            path_to_final_model = extract_data_mixture_and_train(model=model, random_dir=random_dir, tokenizer=tokenizer, 
                                                            train_datasets=train_datasets, 
                                                            val_datasets=val_datasets, 
                                                            data_domains=data_domains, 
                                                            mixing_ratio=input_X[:len(data_domains)], 
                                                            additional_info=all_influences, # add IF value
                                                            total_number_datapoints=total_data, 
                                                            run_name="BO_run_" +str(i),
                                                            method=sampling_method,
                                                            train_epochs=train_epochs, 
                                                            batch_size=training_batch,
                                                            max_step=max_steps,
                                                            lora_config=lora_config,
                                                            eval_steps=eval_steps, 
                                                            callback=[time_callback, elapsed_callback, json_logger_callback],
                                                            seed=seed)
            # free gpu memory
            with torch.no_grad():
                torch.cuda.empty_cache()
            print("evaluating...")
            lora_path = path_to_final_model
            config = PeftConfig.from_pretrained(lora_path)
            model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype='auto')
            # lora_model = PeftModel.from_pretrained(model, lora_path).to(evaluation_cuda)
            lora_model = PeftModel.from_pretrained(model, lora_path)
            tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True,)
            
            observed_performance = 0
            tasks = list(evaluation_task.keys())
            results=evaluate_tasks(tasks, lora_model, tokenizer, evaluation_batch,few_shot=1, limit=limit)
            print("deleting lora model after evaluation.")
            base_path = lora_path.rsplit('/', 1)[0] + '/'
            shutil.rmtree(base_path, ignore_errors=True)
            for task in evaluation_task:
                task_weight, metric = evaluation_task[task]
                if task == "ai2_arc":
                    perf = results["results"]["arc_challenge"][metric]
                else:
                    perf = results["results"][task][metric]
                if task == "wikitext":
                    perf = - perf # we want to maximize the score, so for perplexity we maximize instead
                observed_performance += (perf * task_weight)
            lora_model.to("cpu")
        print("current iteration weighted performance: ", observed_performance)
        # format the observed performance and current parameters for this round with previously seen values
        # see BO tutorial - this does the exact same thing.
        # Notice our BO and GP works with input_X_between_0_1, and not input_X
        current_gp_input = list(input_X_between_0_1)

        # Save results for this iteration
        max_performance_so_far = max(max_performance_so_far, observed_performance)

        results_list.append({
            "iteration": i + 1,
            "mixing_ratio": [to_serializable(x) for x in input_X[:len(data_domains)]], # mixing ratio is the first len(data_domains) elements
            "model_params": [to_serializable(x) for x in input_X[len(data_domains):]], # model params are the last 8 elements
            "performance": float(observed_performance),
            "max_performance_so_far": float(max_performance_so_far)
        })
        # Write to JSON after each iteration for safety
        with open(results_path, "w") as f:
            json.dump(results_list, f, indent=2)
        
        # append observation to a list of historical observation
        GP_input.append(current_gp_input)
        observed_output.append(observed_performance)
        
        # fit the GP with previous observations and observed performance from this round
        gp = SingleTaskGP(torch.DoubleTensor(GP_input), torch.DoubleTensor(observed_output).reshape(-1,1), outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        
        # sanity check:
        print("GP past observed values (should be between [0,1]): ", GP_input)
        
        # use Bayesian Optimization's acq function to propose next candidate mixing parameter and score parameters for agents
        UCB = UpperConfidenceBound(gp, beta=ucb_beta) # the acq function
        #logNEI = LogExpectedImprovement(model=gp, best_f=max(observed_output)) # this is another acq function; ignore for now.
        A = [1.0] * len(data_domains)
        x = list(range(len(data_domains))) # A, x is passed as equality constraints for data mixture. since the ratio needs to sum to 1.
        
        # acq optimization tells us next candidate
        candidate, acq_value = optimize_acqf(
            UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=10,
            equality_constraints = [(torch.tensor(x), torch.tensor(A), 1)] # edit this TODO.
        )
        
        # next candidate are between [0,1] values.
        # We need to perform some reverse engineering to make them into the correct values
        # i.e., reverse normalization.
        def process_values(values, data_domains_len):
            result = []
            
            # Step 1: Squash first `data_domains_len` elements if less than 0.05
            for v in values[:data_domains_len]:
                result.append(0 if v.item() < 0.05 else v)

            # Step 1.5: Normalize the first `data_domains_len` elements to sum to 1
            sum_values = sum(result)
            if sum_values > 0:
                result = [v / sum_values for v in result]
            
            # Step 2: lora layers
            if len(values) > data_domains_len:
                result.append(round(lora_max_num_layers*values[data_domains_len].item()))
            
            # Step 3: Round the next 5 elements: integer options
            start = data_domains_len + 1
            for v in values[start:start+5]:
                result.append(round(v.item()))
            
            # Step 4: lora rank
            if len(values) > start + 5:
                result.append(round(lora_rank_max * values[start + 5].item()))
            
            # Step 5: drop out; unchanged
            if len(values) > start + 6:
                result.append(values[start + 6].item())

            # Step 6: lora alpha
            if len(values) > start + 7:
                result.append(round(lora_rank_max * 2 * values[start + 7].item()))

            print("proposed candidate after processing:", result)
            return result
        print("proposed candidate before processing:", candidate[0])
        
        # these are updated with the candidates and used in next iteration
        input_X_between_0_1 = list(candidate[0])
        input_X = process_values(candidate[0], len(data_domains))
        
    return GP_input, observed_output, gp

def joint_opt_BO_LLM_interleave(time_callback, lora_rank_max, data_domains : List[str], random_dir : str, BO_run : int, total_data : int, evaluation_cuda : str, evaluation_task : dict, ucb_beta, trial_number, sampling_method = "random", train_epochs : int = 1, training_batch : int = 8, evaluation_batch : int = 4, printout=True, max_steps = -1, eval_steps=100, limit=100, model_id = "LLM/llama_8b_instruct", seed = 42, output_dir = "results/"):

    def process_values(values, data_domains_len):
            result = []
            
            # Step 1: Squash first `data_domains_len` elements if less than 0.05
            for v in values[:data_domains_len]:
                result.append(0 if v.item() < 0.05 else v)

            # Step 1.5: Normalize the first `data_domains_len` elements to sum to 1
            sum_values = sum(result)
            if sum_values > 0:
                result = [v / sum_values for v in result]
            
            # Step 2: lora layers
            if len(values) > data_domains_len:
                result.append(round(lora_max_num_layers*values[data_domains_len].item()))
            
            # Step 3: Round the next 5 elements: integer options
            start = data_domains_len + 1
            for v in values[start:start+5]:
                result.append(round(v.item()))
            
            # Step 4: lora rank
            if len(values) > start + 5:
                result.append(round(lora_rank_max * values[start + 5].item()))
            
            # Step 5: drop out; unchanged
            if len(values) > start + 6:
                result.append(values[start + 6].item())
            print("proposed candidate after processing:", result)
            return result
        
    train_datasets = []
    val_datasets = []
    for data_domain in data_domains:
        train_dataset, val_dataset = load_data(data_domain=data_domain)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    # get tokenizer and model
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer, model = get_tokenizer_and_model(model_id = model_id)
    lora_max_num_layers = len(model.model.layers)

    data_input_X = (len(data_domains))*[float(1/len(data_domains))] # initial X is balanced all
    data_input_X_between_0_1 = (len(data_domains))*[float(1/len(data_domains))]

    # lora number of layers - use half the layers for first iteration
    model_input_X = [(int(lora_max_num_layers*0.5))]
    model_input_X_between_0_1 = [0.5]
    
    # apply lora to all modules for first iteration
    model_input_X = model_input_X + [1, 1, 1, 1, 1] # 5 dimension vector to indicate apply to all layers as initial input
    model_input_X_between_0_1 = model_input_X_between_0_1 + [1, 1, 1, 1, 1]
    
    # lora rank of 72 for first iteration
    model_input_X.append(72)
    model_input_X_between_0_1.append(72.0/lora_rank_max)
    
    # lora dropout of 0.05 for first iteration
    model_input_X.append(0.05)
    model_input_X_between_0_1.append(0.05)
    
    # next, define bounds for BO (which interval should our values lie)
    # Recall that BO operates with input_X_between_0_1, which squashed everything to be in [0,1]
    # mixing ratio bounds
    data_lower_bound = [0] * (len(data_domains))
    data_upper_bound = [1] * (len(data_domains))
    
    # lora number of layers bounds
    model_lower_bound = [0]
    model_upper_bound = [1]
    
    # which layer to apply to bounds
    model_lower_bound+=[0, 0, 0, 0, 0]
    model_upper_bound+=[1, 1, 1, 1, 1]
    
    # lora rank bounds
    model_lower_bound.append(0)
    model_upper_bound.append(1)
    
    # lora dropout bounds; this one is not in [0,1] but in [0,0.1]
    model_lower_bound.append(0.0)
    model_upper_bound.append(0.1)
    
    # the actual bounds
    data_bounds = torch.stack([torch.tensor(data_lower_bound), torch.tensor(data_upper_bound)])
    model_bounds = torch.stack([torch.tensor(model_lower_bound), torch.tensor(model_upper_bound)])
    
    GP_input = []
    observed_output = [] 

    all_influences = [] # not used currently
    for train_domain in data_domains:
        all_influences.append(None)
    
    results_list = []
    max_performance_so_far = float('-inf')
    dataset = "_".join(evaluation_task.keys())
    run_BO_on = "interleave"
    results_dir = f"{output_dir}/{dataset}/{run_BO_on}"
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"trial_{trial_number + 1}.json")
    meta_info = {
        "seed": seed,
    }
    meta_path = os.path.join(results_dir, f"trial_{trial_number + 1}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta_info, f, indent=2)

    BO_method = "data"  # For each BO iteration, we either fixed data or model, and optimize other. Interleave between the two. We first fix model and optimize data.
    count = 0

    for i in tqdm(range(BO_run)):
        torch.manual_seed(seed)
        np.random.seed(seed)

        print("iteration: ", i)
        print("data_input_X: ", data_input_X)
        print("model_input_X: ", model_input_X)
        
        # take the model related inputs and arrange them in a nice lora config file
        lora_config = arrange_lora_config(model_input_X[-2], model_input_X[-1], model_input_X[0], model_input_X[1:6])
        
        if lora_config is None:
            observed_performance = -1e6 # very bad performance if we use this
        else: # sample from each domain and train a model
            path_to_final_model = extract_data_mixture_and_train(model=model, random_dir=random_dir, tokenizer=tokenizer, 
                                                            train_datasets=train_datasets, 
                                                            val_datasets=val_datasets, 
                                                            data_domains=data_domains, 
                                                            mixing_ratio=data_input_X, 
                                                            additional_info=all_influences, # add IF value
                                                            total_number_datapoints=total_data, 
                                                            run_name="BO_run_" +str(i),
                                                            method=sampling_method,
                                                            train_epochs=train_epochs, 
                                                            batch_size=training_batch,
                                                            max_step=max_steps,
                                                            lora_config=lora_config,
                                                            eval_steps=eval_steps, 
                                                            callback=[time_callback],
                                                            seed=seed)
            # free gpu memory
            with torch.no_grad():
                torch.cuda.empty_cache()
            print("evaluating...")
            lora_path = path_to_final_model
            config = PeftConfig.from_pretrained(lora_path)
            model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype='auto')
            lora_model = PeftModel.from_pretrained(model, lora_path).to(evaluation_cuda)
            tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True,)
            
            observed_performance = 0
            tasks = list(evaluation_task.keys())
            results=evaluate_tasks(tasks, lora_model, tokenizer, evaluation_batch,few_shot=1, limit=limit)
            print("deleting lora model after evaluation.")
            base_path = lora_path.rsplit('/', 1)[0] + '/'
            shutil.rmtree(base_path, ignore_errors=True)
            for task in evaluation_task:
                task_weight, metric = evaluation_task[task]
                if task == "ai2_arc":
                    perf = results["results"]["arc_challenge"][metric]
                else:
                    perf = results["results"][task][metric]
                if task == "wikitext":
                    perf = - perf # we want to maximize the score, so for perplexity we maximize instead
                observed_performance += (perf * task_weight)
            lora_model.to("cpu")
        print("current iteration weighted performance: ", observed_performance)

        # Save results for this iteration
        max_performance_so_far = max(max_performance_so_far, observed_performance)
        if max_performance_so_far == observed_performance:
            best_data_input_X, best_data_input_X_between_0_1 = data_input_X, data_input_X_between_0_1
            best_model_input_X, best_model_input_X_between_0_1 = model_input_X, model_input_X_between_0_1
        results_list.append({
            "iteration": i + 1,
            "mixing_ratio": str([to_serializable(x) for x in data_input_X]), # mixing ratio is the first len(data_domains) elements
            "model_params": str([to_serializable(x) for x in model_input_X]), # model params are the last 8 elements
            "performance": float(observed_performance),
            "max_performance_so_far": float(max_performance_so_far)
        })
        # Write to JSON after each iteration for safety
        with open(results_path, "w") as f:
            json.dump(results_list, f, indent=2)

        # format the observed performance and current parameters for this round with previously seen values
        # see BO tutorial - this does the exact same thing.
        # Notice our BO and GP works with input_X_between_0_1, and not input_X
        current_gp_input = data_input_X_between_0_1 + model_input_X_between_0_1
        observed_output.append(observed_performance)
        GP_input.append(current_gp_input)
        
        # alternate between optimizing data and model for 5 times each, then switch
        count += 1
        if count == 5:
            count = 0
            if BO_method == "data":
                BO_method = "model"
                data_input_X, data_input_X_between_0_1 = best_data_input_X, best_data_input_X_between_0_1
            else:
                BO_method = "data"
                model_input_X, model_input_X_between_0_1 = best_model_input_X, best_model_input_X_between_0_1
        
        # fit the GP with previous observations and observed performance from this round
        gp = SingleTaskGP(torch.DoubleTensor(GP_input), torch.DoubleTensor(observed_output).reshape(-1,1), outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        # sanity check:
        print("GP past observed values (should be between [0,1]): ", GP_input)
        # use Bayesian Optimization's acq function to propose next candidate mixing parameter and score parameters for agents
        UCB = UpperConfidenceBound(gp, beta=ucb_beta) # the acq function    $TODO: Try qUCB
        #logNEI = LogExpectedImprovement(model=gp, best_f=max(observed_output)) # this is another acq function; ignore for now.
        A = [1.0] * len(data_domains)
        x = list(range(len(data_domains))) # A, x is passed as equality constraints for data mixture. since the ratio needs to sum to 1.
        # append observation to a list of historical observation
        if BO_method == "data":
            fixed_features = {index + len(data_domains): value for index, value in enumerate(model_input_X_between_0_1)}
            print("model fixed features:", fixed_features)
            temp_model_bounds = torch.tensor([model_input_X_between_0_1, model_input_X_between_0_1], dtype=torch.float)
            full_bounds = torch.cat((data_bounds, temp_model_bounds), dim=1)
            candidate, acq_value = optimize_acqf(
                UCB, bounds=full_bounds.to(dtype=torch.float), q=1, num_restarts=5, raw_samples=10,
                fixed_features = fixed_features,
                equality_constraints = [(torch.tensor(x, dtype=torch.long), torch.tensor(A, dtype=torch.float), 1.0)]
            )
            print("proposed candidate before processing:", candidate[0])
            # these are updated with the candidates and used in next iteration
            data_input_X_between_0_1 = list(candidate[0][:])[:len(data_domains)]
            processed_value = process_values(candidate[0], len(data_domains))
            data_input_X = processed_value[:len(data_domains)]
        else:
            fixed_features = {index: value for index, value in enumerate(data_input_X_between_0_1)}
            print("data fixed features:", fixed_features)
            temp_data_bounds = torch.tensor([data_input_X_between_0_1, data_input_X_between_0_1], dtype=torch.float)
            full_bounds = torch.cat((temp_data_bounds, model_bounds), dim=1)
            candidate, acq_value = optimize_acqf(
                UCB, bounds=full_bounds.to(dtype=torch.float), q=1, num_restarts=5, raw_samples=10,
                fixed_features = fixed_features
            )
            print("proposed candidate before processing:", candidate[0])
            # these are updated with the candidates and used in next iteration
            model_input_X_between_0_1 = list(candidate[0])[len(data_domains):]
            processed_value = process_values(candidate[0], len(data_domains))
            model_input_X = processed_value[len(data_domains):]
        
    return GP_input, observed_output, gp

def joint_opt_BO_LLM_fixed_feature_list(time_callback, lora_rank_max, data_domains : List[str], random_dir : str, BO_run : int, total_data : int, evaluation_cuda : str, evaluation_task : dict, ucb_beta, sampling_method = "random", train_epochs : int = 1, training_batch : int = 8, evaluation_batch : int = 4, printout=True, max_steps = -1, eval_steps=100, limit=100, model_id = "LLM/llama_8b_instruct"):
    
    train_datasets = []
    val_datasets = []
    for data_domain in data_domains:
        train_dataset, val_dataset = load_data(data_domain=data_domain)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    # get tokenizer and model
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer, model = get_tokenizer_and_model(model_id = model_id)
    lora_max_num_layers = len(model.model.layers)
    # input_X is the input to our GP:
    # first len(data_domains) are the mixing ratio (0 to 1, constrained to sum to 1)
    # next 1 dimension is the number of layers to apply to (integer)
    # next 5 dimension vector to indicate which layer to apply to (0 or 1)
    # then lora rank (integer)
    # then lora dropout (float)
    
    # for discrete BO
    '''
    fixed_features_list (list[dict[int, float]] | None) 
    A list of maps {feature_index: value}.
    The i-th item represents the fixed_feature for the i-th optimization.
    If fixed_features_list is provided, optimize_acqf_mixed is invoked.
    All indices (feature_index) should be non-negative.
    '''

    # All possible combinations of 0 or 1 for 5 dimensions
    combinations = list(product([0, 1], repeat=5))
    d = len(data_domains)
    # Convert each combination into a dictionary
    dict_list = [{i + d + 1: val for i, val in enumerate(combo)} for combo in combinations]
    print("fixed feature list generated:")
    for x in (dict_list):
        print(x)

    fixed_features_list = dict_list
    
    # mixing ratio
    input_X = (len(data_domains))*[float(1/len(data_domains))] # initial X is balanced all
    input_X_between_0_1 = (len(data_domains))*[float(1/len(data_domains))]
    # lora number of layers
    input_X.append(int(lora_max_num_layers*0.5))
    input_X_between_0_1.append(0.5)
    # lora which layer to apply to
    input_X = input_X + [1, 1, 1, 1, 1] # 5 dimension vector to indicate apply to all layers as initial input
    input_X_between_0_1 = input_X_between_0_1 + [1, 1, 1, 1, 1]
    # lora rank
    input_X.append(72) # initial rank = 16
    input_X_between_0_1.append(72.0/lora_rank_max)
    # lora dropout
    input_X.append(0.05) # initial dropout=0.05
    input_X_between_0_1.append(0.05)
    # mixing ratio bounds
    lower_bound = [0] * (len(data_domains))
    upper_bound = [1] * (len(data_domains))
    # lora number of layers bounds
    lower_bound.append(0)
    upper_bound.append(1)
    # which layer to apply to bounds
    lower_bound+=[0, 0, 0, 0, 0]
    upper_bound+=[1, 1, 1, 1, 1]
    # lora rank bounds
    lower_bound.append(0)
    upper_bound.append(1)
    # lora dropout bounds
    lower_bound.append(0.0)
    upper_bound.append(0.1)
    bounds = torch.stack([torch.tensor(lower_bound), torch.tensor(upper_bound)])
    
    GP_input = []
    observed_output = []

    all_influences = []
    for train_domain in data_domains:
        all_influences.append(None)
        #all_influences.append(torch.load("influence/"+str(train_domain)+"_training.pt"))
    
    for i in tqdm(range(BO_run)):
        print("iteration: ", i)
        print("input_X: ", input_X)
        if printout:
            print("mixing data with method: ", sampling_method)
        
        lora_config = arrange_lora_config(input_X[-2], input_X[-1], input_X[len(data_domains)], input_X[len(data_domains)+1:len(data_domains)+6])
        if lora_config is None:
            observed_performance = 0.1 # very bad performance if we use this
        else: # sample from each domain and train a model
            path_to_final_model = extract_data_mixture_and_train(model=model, random_dir=random_dir, tokenizer=tokenizer, 
                                                            train_datasets=train_datasets, 
                                                            val_datasets=val_datasets, 
                                                            data_domains=data_domains, 
                                                            mixing_ratio=input_X[:len(data_domains)], 
                                                            additional_info=all_influences, # add IF value
                                                            total_number_datapoints=total_data, 
                                                            run_name="BO_run_" +str(i),
                                                            method=sampling_method,
                                                            train_epochs=train_epochs, 
                                                            batch_size=training_batch,
                                                            max_step=max_steps,
                                                            lora_config=lora_config,
                                                            eval_steps=eval_steps, callback=[time_callback])
            # free gpu memory
            with torch.no_grad():
                torch.cuda.empty_cache()
            print("evaluating...")
            lora_path = path_to_final_model
            config = PeftConfig.from_pretrained(lora_path)
            model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype='auto')
            lora_model = PeftModel.from_pretrained(model, lora_path).to(evaluation_cuda)
            tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True,)
            
            observed_performance = 0
            tasks = list(evaluation_task.keys())
            results=evaluate_tasks(tasks, lora_model, tokenizer, evaluation_batch,few_shot=1, limit=limit)
            print("deleting lora model after evaluation.")
            shutil.rmtree(lora_path, ignore_errors=True)
            for task in evaluation_task:
                task_weight, metric = evaluation_task[task]
                perf = results["results"][task][metric]
                if task == "wikitext":
                    perf = - perf # we want to maximize the score, so for perplexity we maximize instead
                observed_performance += (perf * task_weight)
            lora_model.to("cpu")
        print("current iteration weighted performance: ", observed_performance)
        # format the observed performance and current parameters for this round with previously seen values
        current_gp_input = list(input_X_between_0_1)
        
        GP_input.append(current_gp_input)
        observed_output.append(observed_performance)
        
        # fit the GP with previous selected parameters and observed performance from this round
        gp = SingleTaskGP(torch.DoubleTensor(GP_input), torch.DoubleTensor(observed_output).reshape(-1,1), outcome_transform=Standardize(m=1))
        print("GP past observed values (should be between [0,1]): ", GP_input)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        
        # use Bayesian Optimization to propose next candidate mixing parameter and score parameters for agents
        UCB = UpperConfidenceBound(gp, beta=ucb_beta)
        #logNEI = LogExpectedImprovement(model=gp, best_f=max(observed_output))
        A = [1.0] * len(data_domains)
        x = list(range(len(data_domains)))
        candidate, acq_value = optimize_acqf_mixed(
            UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=10,
            equality_constraints = [(torch.tensor(x), torch.tensor(A), 1)],
            fixed_features_list = fixed_features_list# edit this TODO.
        )
        
        def process_values(values, data_domains_len):
            result = []
            
            # Step 1: Squash first `data_domains_len` elements if less than 0.05
            for v in values[:data_domains_len]:
                result.append(0 if v.item() < 0.05 else v)
            
            # Step 2: lora layers
            if len(values) > data_domains_len:
                result.append(round(lora_max_num_layers*values[data_domains_len].item()))
            
            # Step 3: Round the next 5 elements: integer options
            start = data_domains_len + 1
            for v in values[start:start+5]:
                result.append(round(v.item()))
            
            # Step 4: lora rank
            if len(values) > start + 5:
                result.append(round(lora_rank_max * values[start + 5].item()))
            
            # Step 5: drop out; unchanged
            if len(values) > start + 6:
                result.append(values[start + 6].item())
            print("proposed candidate after normalizing:", result)
            return result
        print("proposed candidate before processing:", candidate[0])
        input_X_between_0_1 = list(candidate[0])
        input_X = process_values(candidate[0], len(data_domains))
        
    return GP_input, observed_output, gp

def joint_opt_BO_LLM_only_model(time_callback, lora_rank_max, data_domains : List[str], random_dir : str, BO_run : int, total_data : int, evaluation_cuda : str, evaluation_task : dict, ucb_beta, trial_number, sampling_method = "random", train_epochs : int = 1, training_batch : int = 8, evaluation_batch : int = 4, printout=True, max_steps = -1, eval_steps=100, limit=100, model_id = "LLM/llama_8b_instruct", seed = 42, output_dir = "results/"):

    train_datasets = []
    val_datasets = []
    for data_domain in data_domains:
        train_dataset, val_dataset = load_data(data_domain=data_domain)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    # get tokenizer and model
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer, model = get_tokenizer_and_model(model_id = model_id)
    lora_max_num_layers = len(model.model.layers)

    # input_X is the input to our GP:
    # first len(data_domains) are the mixing ratio (0 to 1, constrained to sum to 1)
    # next 1 dimension is the number of layers to apply to (integer)
    # next 5 dimension vector to indicate which layer to apply to (0 or 1)
    # then lora rank (integer)
    # then lora dropout (float)
    
    # for discrete BO
    fixed_features_list =[{len(data_domains)+2:0},{len(data_domains)+2:1},
                          {len(data_domains)+3:0},{len(data_domains)+3:1},
                          {len(data_domains)+4:0},{len(data_domains)+4:1},
                          {len(data_domains)+5:0},{len(data_domains)+5:1},
                          {len(data_domains)+6:0},{len(data_domains)+6:1}]
    
    # mixing ratio
    input_X = []
    input_X_between_0_1 = []
    lower_bound = []
    upper_bound = []

    # def get_mixing_ratio(evaluation_task):
    #     dataset = "_".join(evaluation_task.keys())
    #     if dataset == "gsm8k":
    #         return [0,0,0.14,0.31,0.12,0.14,0,0.29,0,0]
    #     elif dataset == "commonsense_qa":
    #         return [0,0,0,1,0,0,0,0,0,0]
    #     elif dataset == "headqa_en":
    #         return [0.1221754401922226,0.0,0.539222776889801,0.0,0.0,0.0,0.2574373185634613,0.0,0.0,0.0811644196510315]
    #     elif dataset == "pubmedqa":
    #         return [0.0,0.0,0.0,0.0,0.0,0.06087161973118782,0.9391283392906189,0.0,0.0,0.0]
    #     elif dataset == "triviaqa":
    #         return [0.0,0.0,0.0,0.6801438927650452,0.11240127682685852,0.0,0.2074548453092575,0.0,0.0,0.0]
    #     elif dataset == "truthfulqa_gen":
    #         return [0.0,0.0,0.0,0.0,0.0,0.20507599413394928,0.0,0.0,0.0,0.7949240207672119]
    #     else:   # For wikitext, mmlu, ai2_arc: no bad mixing ratio found, so we use arbitrary bad mixing ratio
    #         return [1,0,0,0,0,0,0,0,0,0]

    # mixing_ratio = get_mixing_ratio(evaluation_task)
    mixing_ratio = (len(data_domains))*[float(1/len(data_domains))] # initial X is balanced all

    # lora number of layers
    input_X.append(int(lora_max_num_layers*0.5))
    input_X_between_0_1.append(0.5)
    # lora which layer to apply to
    input_X = input_X + [1, 1, 1, 1, 1] # 5 dimension vector to indicate apply to all layers as initial input
    input_X_between_0_1 = input_X_between_0_1 + [1, 1, 1, 1, 1]
    # lora rank
    input_X.append(72) # initial rank = 72
    input_X_between_0_1.append(72.0/lora_rank_max)
    # lora dropout
    input_X.append(0.05) # initial dropout=0.05
    input_X_between_0_1.append(0.05)

    # lora number of layers bounds
    lower_bound.append(0)
    upper_bound.append(1)
    # which layer to apply to bounds
    lower_bound+=[0, 0, 0, 0, 0]
    upper_bound+=[1, 1, 1, 1, 1]
    # lora rank bounds
    lower_bound.append(0)
    upper_bound.append(1)
    # lora dropout bounds
    lower_bound.append(0.0)
    upper_bound.append(0.1)
    bounds = torch.stack([torch.tensor(lower_bound), torch.tensor(upper_bound)])
    
    GP_input = []
    observed_output = []

    all_influences = []
    for train_domain in data_domains:
        all_influences.append(None)
        #all_influences.append(torch.load("influence/"+str(train_domain)+"_training.pt"))\

    results_list = []
    max_performance_so_far = float('-inf')
    dataset = "_".join(evaluation_task.keys())
    run_BO_on = "model"
    results_dir = f"{output_dir}/{dataset}/{run_BO_on}"
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"trial_{trial_number + 1}.json")
    meta_info = {
        "seed": seed,
        "initial_mixing_ratio": mixing_ratio,
    }
    meta_path = os.path.join(results_dir, f"trial_{trial_number + 1}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta_info, f, indent=2)
        

    for i in tqdm(range(BO_run)):
        torch.manual_seed(seed)
        np.random.seed(seed)

        # get tokenizer and model
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        tokenizer, model = get_tokenizer_and_model(model_id = model_id)

        print("iteration: ", i)
        print("input_X: ", input_X)
        if printout:
            print("mixing data with method: ", sampling_method)
        # lora_r, lora_dropout, num_layers_to_apply, five_dim_vector
        lora_config = arrange_lora_config(input_X[6], input_X[7], input_X[0], input_X[1:6])

        if lora_config is None:
            observed_performance = -1e6 # very bad performance if we use this
        else: # sample from each domain and train a model
            path_to_final_model = extract_data_mixture_and_train(model=model, random_dir=random_dir, tokenizer=tokenizer, 
                                                            train_datasets=train_datasets, 
                                                            val_datasets=val_datasets, 
                                                            data_domains=data_domains, 
                                                            mixing_ratio=mixing_ratio, 
                                                            additional_info=all_influences, # add IF value
                                                            total_number_datapoints=total_data, 
                                                            run_name="BO_run_" +str(i),
                                                            method=sampling_method,
                                                            train_epochs=train_epochs, 
                                                            batch_size=training_batch,
                                                            max_step=max_steps,
                                                            lora_config=lora_config,
                                                            eval_steps=eval_steps, 
                                                            callback=[time_callback],
                                                            seed=seed)
            # free gpu memory
            with torch.no_grad():
                torch.cuda.empty_cache()
            print("evaluating...")
            lora_path = path_to_final_model
            config = PeftConfig.from_pretrained(lora_path)
            model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype='auto')
            lora_model = PeftModel.from_pretrained(model, lora_path).to(evaluation_cuda)
            tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True,)
            
            observed_performance = 0
            tasks = list(evaluation_task.keys())
            results=evaluate_tasks(tasks, lora_model, tokenizer, evaluation_batch,few_shot=1, limit=limit)
            print("deleting lora model after evaluation.")
            base_path = lora_path.rsplit('/', 1)[0] + '/'
            shutil.rmtree(base_path, ignore_errors=True)
            for task in evaluation_task:
                task_weight, metric = evaluation_task[task]
                if task == "ai2_arc":
                    perf = results["results"]["arc_challenge"][metric]
                else:
                    perf = results["results"][task][metric]
                if task == "wikitext":
                    perf = - perf # we want to maximize the score, so for perplexity we maximize instead
                observed_performance += (perf * task_weight)
            lora_model.to("cpu")
        print("current iteration weighted performance: ", observed_performance)
        # format the observed performance and current parameters for this round with previously seen values
        current_gp_input = list(input_X_between_0_1)

        # Save results for this iteration
        max_performance_so_far = max(max_performance_so_far, observed_performance)
        results_list.append({
            "iteration": i + 1,
            "model_params": list(input_X),
            "performance": observed_performance,
            "max_performance_so_far": max_performance_so_far
        })
        # Write to JSON after each iteration for safety
        with open(results_path, "w") as f:
            json.dump(results_list, f, indent=2)
        
        GP_input.append(current_gp_input)
        observed_output.append(observed_performance)
        
        # fit the GP with previous selected parameters and observed performance from this round
        gp = SingleTaskGP(torch.DoubleTensor(GP_input), torch.DoubleTensor(observed_output).reshape(-1,1), outcome_transform=Standardize(m=1))
        print("GP past observed values (should be between [0,1]): ", GP_input)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        
        # use Bayesian Optimization to propose next candidate mixing parameter and score parameters for agents
        UCB = UpperConfidenceBound(gp, beta=ucb_beta)
        #logNEI = LogExpectedImprovement(model=gp, best_f=max(observed_output))
        A = [1.0] * len(data_domains)
        x = list(range(len(data_domains)))
        candidate, acq_value = optimize_acqf(
            UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=10,
            # equality_constraints = [(torch.tensor(x, dtype=torch.float), torch.tensor(A, dtype=torch.float), 1)] # remove this line because we do not use data mixture here
        )
        
        def process_values(values):
            result = []
            
            # Step 1: Squash first `data_domains_len` elements if less than 0.05
            
            # Step 2: lora layers
            result.append(round(lora_max_num_layers*values[0].item()))
            
            # Step 3: Round the next 5 elements: integer options
            for v in values[1:6]:
                result.append(round(v.item()))
            
            # Step 4: lora rank
            result.append(round(lora_rank_max * values[6].item()))
            
            # Step 5: drop out; unchanged
            result.append(values[7].item())
            print("proposed candidate after normalizing:", result)
            return result
        print("proposed candidate before processing:", candidate[0])
        input_X_between_0_1 = list(candidate[0])
        input_X = process_values(candidate[0])
        
    return GP_input, observed_output, gp

def joint_opt_random(time_callback, lora_rank_max, data_domains : List[str], random_dir : str, BO_run : int, total_data : int, evaluation_cuda : str, evaluation_task : dict, ucb_beta, sampling_method = "random", train_epochs : int = 1, training_batch : int = 8, evaluation_batch : int = 4, printout=True, max_steps = -1, eval_steps=100, limit=100, model_id = "LLM/llama_8b_instruct"):

    train_datasets = []
    val_datasets = []
    for data_domain in data_domains:
        train_dataset, val_dataset = load_data(data_domain=data_domain)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    # get tokenizer and model
    tokenizer, model = get_tokenizer_and_model(model_id = model_id)
    lora_max_num_layers = len(model.model.layers)
    
    # input_X is the input to our GP:
    # first len(data_domains) are the mixing ratio for data (0 to 1, constrained to sum to 1)
    # next 1 dimension is the number of layers to apply to (integer)
    # next 5 dimension vector to indicate which layer to apply to (0 or 1)
    # then lora rank (integer)
    # then lora dropout (float)
    
    # input_X_between_0_1 is the standardized form of input_X (with everything between 0 and 1)
    # We use this input for the BO to make optimization more stable.
    
    # The following represents the inputs used for first iteration, so it's hard coded.
    # mixing ratio - even ratio for all domains for first iteration
    input_X = (len(data_domains))*[float(1/len(data_domains))] # initial X is balanced all
    input_X_between_0_1 = (len(data_domains))*[float(1/len(data_domains))]
    
    # lora number of layers - use half the layers for first iteration
    input_X.append(int(lora_max_num_layers*0.5))
    input_X_between_0_1.append(0.5)
    
    # apply lora to all modules for first iteration
    input_X = input_X + [1, 1, 1, 1, 1] # 5 dimension vector to indicate apply to all layers as initial input
    input_X_between_0_1 = input_X_between_0_1 + [1, 1, 1, 1, 1]
    
    # lora rank of 72 for first iteration
    input_X.append(72)
    input_X_between_0_1.append(72.0/lora_rank_max)
    
    # lora dropout of 0.05 for first iteration
    input_X.append(0.05)
    input_X_between_0_1.append(0.05)
    
    # next, define bounds for BO (which interval should our values lie)
    # Recall that BO operates with input_X_between_0_1, which squashed everything to be in [0,1]
    # mixing ratio bounds
    lower_bound = [0] * (len(data_domains))
    upper_bound = [1] * (len(data_domains))
    
    # lora number of layers bounds
    lower_bound.append(0)
    upper_bound.append(1)
    
    # which layer to apply to bounds
    lower_bound+=[0, 0, 0, 0, 0]
    upper_bound+=[1, 1, 1, 1, 1]
    
    # lora rank bounds
    lower_bound.append(0)
    upper_bound.append(1)
    
    # lora dropout bounds; this one is not in [0,1] but in [0,0.1]
    lower_bound.append(0.0)
    upper_bound.append(0.1)
    
    # the actual bounds
    bounds = torch.stack([torch.tensor(lower_bound), torch.tensor(upper_bound)])
    
    GP_input = [] # X
    observed_output = [] # y

    all_influences = [] # not used currently
    for train_domain in data_domains:
        all_influences.append(None)
        #all_influences.append(torch.load("influence/"+str(train_domain)+"_training.pt"))
    
    # for each BO iteration, do this...
    for i in tqdm(range(BO_run)):
        print("iteration: ", i)
        print("input_X: ", input_X)
        if printout:
            print("mixing data with method: ", sampling_method)
        
        # take the model related inputs and arrange them in a nice lora config file
        lora_config = arrange_lora_config(input_X[-2], input_X[-1], input_X[len(data_domains)], input_X[len(data_domains)+1:len(data_domains)+6])
        
        if lora_config is not None:
            
            # sample from each domain and train a model according to data mixture ratio
            # and the chosen lora config file which determines the model architecture
            # path_to_final_model is the path to the trained model
            path_to_final_model = extract_data_mixture_and_train(model=model, random_dir=random_dir, tokenizer=tokenizer, 
                                                            train_datasets=train_datasets, 
                                                            val_datasets=val_datasets, 
                                                            data_domains=data_domains, 
                                                            mixing_ratio=input_X[:len(data_domains)], 
                                                            additional_info=all_influences, # not used atm
                                                            total_number_datapoints=total_data, 
                                                            run_name="BO_run_" +str(i),
                                                            method=sampling_method,
                                                            train_epochs=train_epochs, 
                                                            batch_size=training_batch,
                                                            max_step=max_steps,
                                                            lora_config=lora_config,
                                                            eval_steps=eval_steps, callback=[time_callback])
            # free the gpu memory
            with torch.no_grad():
                torch.cuda.empty_cache()
            
            # load the model from path_to_final_model
            print("evaluating...")
            lora_path = path_to_final_model
            config = PeftConfig.from_pretrained(lora_path)
            model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype='auto')
            lora_model = PeftModel.from_pretrained(model, lora_path).to(evaluation_cuda)
            tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True,)
            
            # ideally we only have one evaluation task. But the code below works
            # for any weighted average of several task. But for now, we only use a single task.
            # each task has a specified metric that's passed here.
            observed_performance = 0
            tasks = list(evaluation_task.keys())
            results=evaluate_tasks(tasks, lora_model, tokenizer, evaluation_batch,few_shot=1, limit=limit)
            for task in evaluation_task:
                task_weight, metric = evaluation_task[task]
                perf = results["results"][task][metric]
                if task == "wikitext":
                    perf = - perf # we want to maximize the score, so for wikitext perplexity we maximize instead
                observed_performance += (perf * task_weight)
            lora_model.to("cpu")

            print("deleting lora model after evaluation.") # after evaluation, delete the model since no need already.
            shutil.rmtree(lora_path, ignore_errors=True)
        
        else:
            observed_performance = 0.1
        
        print("current iteration weighted performance: ", observed_performance)
        # generate random candidate:
        #[tensor(0.2207), tensor(0.2730), tensor(0.0525), tensor(0.2114), 0,
        # tensor(0.1078), 0, tensor(0.1324), 10, 0, 0, 0, 0, 1, 34, 0.0748564749956131]
        # length is len_domain + 1 + 5 + 1 + 1
        def random_generator(data_domains, num_extra_vals=3, max_value=100):
            result = []

            # a) First len(data_domains) values sum to 1
            weights = np.random.dirichlet(np.ones(len(data_domains))).tolist()
            result.extend(weights)

            # b) One random value between 0 and 1
            result.append(random.uniform(0, 1))

            # c) Next 5 values are either 0 or 1
            result.extend([random.randint(0, 1) for _ in range(5)])

            # d) Next num_extra_vals random values between 0 and 1
            result.extend([random.uniform(0, 1) for _ in range(num_extra_vals)])

            # e) Last value is between 0 and max_value
            result.append(random.uniform(0, max_value))


            return result

        candidate = [random_generator(data_domains, 5, 0.1)]
        
        # next candidate are between [0,1] values.
        # We need to perform some reverse engineering to make them into the correct values
        # i.e., reverse normalization.
        def process_values(values, data_domains_len):
            result = []
            
            # Step 1: Squash first `data_domains_len` elements if less than 0.05
            for v in values[:data_domains_len]:
                result.append(0 if v.item() < 0.05 else v)
            
            # Step 2: lora layers
            if len(values) > data_domains_len:
                result.append(round(lora_max_num_layers*values[data_domains_len].item()))
            
            # Step 3: Round the next 5 elements: integer options
            start = data_domains_len + 1
            for v in values[start:start+5]:
                result.append(round(v.item()))
            
            # Step 4: lora rank
            if len(values) > start + 5:
                result.append(round(lora_rank_max * values[start + 5].item()))
            
            # Step 5: drop out; unchanged
            if len(values) > start + 6:
                result.append(values[start + 6].item())
            print("proposed candidate after processing:", result)
            return result
        print("proposed candidate before processing:", candidate[0])
        
        current_gp_input = list(input_X_between_0_1)
        GP_input.append(current_gp_input)
        observed_output.append(observed_performance)
        
        # these are updated with the candidates and used in next iteration
        input_X_between_0_1 = list(candidate[0])
        input_X = process_values(candidate[0], len(data_domains))
        
    return GP_input, observed_output

def evaluate_single_configuration(time_callback, lora_rank_max, data_domains : List[str], random_dir : str, total_data : int, evaluation_cuda : str, evaluation_task : dict, seed : int, init_mixing_ratio: List[float] = None, init_lora_num_layers: int = None, init_lora_modules: List[int] = None, init_lora_rank: int = None, init_lora_dropout: float = None, sampling_method = "random", train_epochs : int = 1, training_batch : int = 8, evaluation_batch : int = 4, max_steps = -1, eval_steps=100, limit=100, model_id = "LLM/llama_8b_instruct", iteration=1, trial_idx=1, output_dir="results/") -> float:

    def initialise_values(init_mixing_ratio, init_lora_num_layers, init_lora_modules, init_lora_rank, init_lora_dropout):
        if init_mixing_ratio is None:
            input_X = (len(data_domains))*[float(1/len(data_domains))]
            input_X_between_0_1 = (len(data_domains))*[float(1/len(data_domains))]
        else:
            input_X = init_mixing_ratio.copy()
            input_X_between_0_1 = init_mixing_ratio.copy()
        if init_lora_num_layers is None:
            input_X.append(16)
            input_X_between_0_1.append(0.5)
        else:
            input_X.append(init_lora_num_layers)
            input_X_between_0_1.append(init_lora_num_layers/32)  # assuming max layers is 32, hard code for now
        if init_lora_modules is None:
            input_X = input_X + [1, 1, 1, 1, 1] # apply lora to all modules
            input_X_between_0_1 = input_X_between_0_1 + [1, 1, 1, 1, 1] # apply lora to all modules
        else:
            input_X = input_X + init_lora_modules
            input_X_between_0_1 = input_X_between_0_1 + init_lora_modules
        if init_lora_rank is None:
            input_X.append(72)
            input_X_between_0_1.append(72.0/lora_rank_max)
        else:
            input_X.append(init_lora_rank)
            input_X_between_0_1.append(init_lora_rank/lora_rank_max)
        if init_lora_dropout is None:
            input_X.append(0.05)
            input_X_between_0_1.append(0.05)
        else:
            input_X.append(init_lora_dropout)
            input_X_between_0_1.append(init_lora_dropout)
        
        return input_X, input_X_between_0_1

    # input_X, input_X_between_0_1 = initialise_values(init_mixing_ratio, init_lora_num_layers, init_lora_modules, init_lora_rank, init_lora_dropout)

    # def initialise_values(init_mixing_ratio=None, 
    #                   init_lora_num_layers=None, 
    #                   init_lora_modules=None, 
    #                   init_lora_rank=None, 
    #                   init_lora_dropout=None,
    #                   lora_rank_max=128,   # default max rank
    #                   data_domains=None):  # assuming you pass this if needed

    #     # ---------- Mixing ratio ----------
    #     if init_mixing_ratio is None:
    #         # generate 10 random values summing to 1
    #         rand_vals = np.random.rand(10)
    #         input_X_mixing = (rand_vals / rand_vals.sum()).tolist()
    #         input_X_between_0_1_mixing = input_X_mixing.copy()
    #     else:
    #         input_X_mixing = init_mixing_ratio.copy()
    #         input_X_between_0_1_mixing = init_mixing_ratio.copy()
        
    #     # ---------- LoRA num layers ----------
    #     if init_lora_num_layers is None:
    #         input_X_num_layers = random.randint(0, 31)
    #         input_X_between_0_1_num_layers = input_X_num_layers / 32  # normalize
    #     else:
    #         input_X_num_layers = init_lora_num_layers
    #         input_X_between_0_1_num_layers = init_lora_num_layers / 32
        
    #     # ---------- LoRA modules ----------
    #     if init_lora_modules is None:
    #         input_X_modules = [random.randint(0, 1) for _ in range(5)]
    #         input_X_between_0_1_modules = input_X_modules.copy()
    #     else:
    #         input_X_modules = init_lora_modules.copy()
    #         input_X_between_0_1_modules = init_lora_modules.copy()
        
    #     # ---------- LoRA rank ----------
    #     if init_lora_rank is None:
    #         input_X_rank = random.randint(0, 127)
    #         input_X_between_0_1_rank = input_X_rank / lora_rank_max
    #     else:
    #         input_X_rank = init_lora_rank
    #         input_X_between_0_1_rank = init_lora_rank / lora_rank_max
        
    #     # ---------- LoRA dropout ----------
    #     if init_lora_dropout is None:
    #         input_X_dropout = random.uniform(0, 1)
    #         input_X_between_0_1_dropout = input_X_dropout
    #     else:
    #         input_X_dropout = init_lora_dropout
    #         input_X_between_0_1_dropout = init_lora_dropout
        
    #     # ---------- Combine all ----------
    #     input_X = input_X_mixing + [input_X_num_layers] + input_X_modules + [input_X_rank, input_X_dropout]
    #     input_X_between_0_1 = input_X_between_0_1_mixing + [input_X_between_0_1_num_layers] + \
    #                         input_X_between_0_1_modules + [input_X_between_0_1_rank, input_X_between_0_1_dropout]
        
    #     return input_X, input_X_between_0_1

    # def initialise_values_5000(trial_idx, evaluation_task):
    #     task = "_".join(evaluation_task.keys())
    #     file_path = f"results_extrapolation/{task}/run_{trial_idx}/performance_50.json"
    #     if os.path.exists(file_path):
    #         with open(file_path, "r") as f:
    #             data = json.load(f)
    #         input_X = data[0]["params"]
    #         seed = data[0]["seed"]
    #         return input_X, seed
    #     else:
    #         return None, None

    # if total_data != 50:
    #     # For 5000 data points, we use the initial values from the 1000 data points run
    #     input_X, seed = initialise_values_5000(trial_idx, evaluation_task)
    # else:
    #     # For other data points, we initialize randomly
    #     input_X, input_X_between_0_1 = initialise_values()

    # print("initial input_X: ", input_X)
    print("seed:", seed)
    
    # train_datasets = []
    # val_datasets = []
    # for data_domain in data_domains:
    #     train_dataset, val_dataset = load_data(data_domain=data_domain)
    #     train_datasets.append(train_dataset)
    #     val_datasets.append(val_dataset)

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # get tokenizer and model
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer, model = get_tokenizer_and_model(model_id = model_id)
    
    # Create LoRA config
    # lora_config = arrange_lora_config(
    #     input_X[-2],                         # lora rank
    #     input_X[-1],                         # dropout
    #     input_X[len(data_domains)],         # lora layer
    #     input_X[len(data_domains)+1:len(data_domains)+6]  # other lora params
    # )

    # all_influences = [] # not used currently
    # for train_domain in data_domains:
    #     all_influences.append(None)

    # if lora_config is None:
    #     return -1e6  # penalise bad confi

    # # Train model
    # path_to_final_model = extract_data_mixture_and_train(
    #     model=model,
    #     random_dir=random_dir,
    #     tokenizer=tokenizer,
    #     train_datasets=train_datasets,
    #     val_datasets=val_datasets,
    #     data_domains=data_domains,
    #     mixing_ratio=input_X[:len(data_domains)],
    #     additional_info=all_influences,
    #     total_number_datapoints=total_data,
    #     run_name="manual_eval",
    #     method=sampling_method,
    #     train_epochs=train_epochs,
    #     batch_size=training_batch,
    #     max_step=max_steps,
    #     lora_config=lora_config,
    #     eval_steps=eval_steps,
    #     callback=[time_callback] if time_callback else [],
    #     seed=seed,
    #     results_dir=output_dir
    # )

    # # free gpu memory
    # with torch.no_grad():
    #     torch.cuda.empty_cache()
    print("evaluating...")
    # lora_path = path_to_final_model
    # config = PeftConfig.from_pretrained(lora_path)
    # model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype='auto')
    # lora_model = PeftModel.from_pretrained(model, lora_path).to(evaluation_cuda)
    # tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True,)
    
    observed_performance = {}
    tasks = list(evaluation_task.keys())
    model.eval()
    results=evaluate_tasks(tasks, model, tokenizer, evaluation_batch,few_shot=1, limit=limit)
    # print("deleting lora model after evaluation.")
    for task in evaluation_task:
        task_weight, metric = evaluation_task[task]
        if task == "ai2_arc":
            perf = results["results"]["arc_challenge"][metric]
        else:
            perf = results["results"][task][metric]
        if task == "wikitext":
            perf = - perf
        observed_performance[task] = perf
    model.to("cpu")
    print("Observed performance: ", observed_performance)
    with open(f"base_performance/{model_id.replace('/', '_')}.json", "w") as f:
        json.dump(observed_performance, f, indent=2)

    # results_data = {
    #     "seed": seed,
    #     "evaluation_task": tasks,
    #     "params": input_X,
    #     "performance": observed_performance
    # }
    # with open(output_dir + f"/performance_{total_data}.json", "w") as f:
    #     json.dump([results_data], f, indent=2)  

    # print("idx = ", trial_idx)

    # # Save results
    # task = "_".join(evaluation_task.keys())
    # filename = f"results_full_run/{task}/all/trial_{trial_idx}.json"
    # print("Saving results to:", filename)
    # with open(filename, "r") as f:
    #     data = json.load(f)
    
    # for entry in data:
    #     if entry["iteration"] == iteration:
    #         print("Updating entry for iteration:", iteration)
    #         print("Previous performance (after 100s): ", entry["performance"])
    #         print("New performance: ", observed_performance)
    #         entry[f"performance_after_{total_data}"] = observed_performance
    #         entry.pop("performance_after_500", None)    # Delete performance_after_500 if it exists
    #         break
    
    # with open(filename, "w") as f:
    #     json.dump(data, f, indent=2)

    return observed_performance

def sample_random_params(num_points, d):
    out = []
    ranks = np.array([8,16,32,64,128])
    for _ in range(num_points):
        # a) mixing: Dirichlet so sum-to-1
        mix = np.random.dirichlet(np.ones(d))
        # b) num_layers: uniform int between 1 and max_layers(32)
        num_layers = np.random.randint(1, 32+1)
        # c) module_flags: uniform random 0/1
        flags = np.random.randint(0,2,size=5)
        # d) rank: choose uniformly from allowed
        rank = np.random.choice(ranks)
        # e) dropout: uniform in [0, max_dropout]
        dropout = np.random.rand()*0.2
        x = np.concatenate([mix,
                            [num_layers],
                            flags,
                            [rank],
                            [dropout]])
        out.append(x)
    return np.stack(out)  # shape (num_points, 18)

class ParamVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1  = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.fc3  = nn.Linear(latent_dim, hidden_dim)
        self.fc4  = nn.Linear(hidden_dim, input_dim)
        self.register_buffer('scaler_mean', torch.zeros(input_dim))
        self.register_buffer('scaler_std',  torch.ones(input_dim))

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc21(h), self.fc22(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x_norm = self.fc4(F.relu(self.fc3(z)))
        return x_norm * self.scaler_std + self.scaler_mean
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def train_vae(
    x_tensor: torch.Tensor,
    latent_dim: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    device: torch.device
) -> ParamVAE:
    input_dim = x_tensor.size(1)
    mean = x_tensor.mean(dim=0)
    std  = x_tensor.std(dim=0, unbiased=False)
    x_tensor = (x_tensor - mean) / std

    vae = ParamVAE(input_dim, latent_dim, hidden_dim).to(device)
    vae.scaler_mean.copy_(mean.to(device))
    vae.scaler_std.copy_(std.to(device))

    opt = optim.Adam(vae.parameters(), lr=lr)
    x_tensor = x_tensor.to(device)
    for _ in range(epochs):
        recon, mu, logvar = vae(x_tensor)
        recon_norm = (recon - vae.scaler_mean) / vae.scaler_std
        recon_loss = F.mse_loss(recon_norm, x_tensor)
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kld
        opt.zero_grad()
        loss.backward()
        opt.step()
    return vae

def joint_opt_BO_LLM_with_vae(
    time_callback,
    lora_rank_max: int,
    data_domains: List[str],
    random_dir: str,
    BO_run: int,
    total_data: int,
    evaluation_cuda: str,
    evaluation_task: dict,
    ucb_beta: float,
    trial_number: int,
    sampling_method="random",
    train_epochs: int = 1,
    training_batch: int = 8,
    evaluation_batch: int = 4,
    printout=True,
    max_steps: int = -1,
    eval_steps: int = 100,
    limit=100,
    seed = 42, 
    output_dir = "results/",
    model_id="meta-llama/Meta-Llama-3-8B-Instruct",
    latent_dim: int = 4,
    vae_hidden: int = 128,
    vae_epochs: int = 50,
    vae_lr: float = 1e-3,
):
    """
    BO in latent space via a VAE, then decode back to original 18-D parameters.
    """
    
    # 1) Warm-start two iterations of normal BO
    warmup_X, warmup_y, _ = joint_opt_BO_LLM(
        time_callback, lora_rank_max, data_domains, random_dir,
        2, total_data, evaluation_cuda, evaluation_task,
        ucb_beta, trial_number, sampling_method,
        train_epochs, training_batch, evaluation_batch,
        printout, max_steps, eval_steps, limit,
        model_id, seed=seed, output_dir="dummy_output_dir",
    )

    GP_input = [list(x) for x in warmup_X]
    observed_output = list(warmup_y)

    main_vae_dir = os.path.join(os.getcwd(), "vae")
    os.makedirs(main_vae_dir, exist_ok=True)
    vae_path = os.path.join(main_vae_dir, f"vae_latent{latent_dim}_hidden{vae_hidden}.pth")
    scaler_path = os.path.join(main_vae_dir, f"scaler_ld{latent_dim}_hd{vae_hidden}.npz")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = len(data_domains)
    vae_path = os.path.join(main_vae_dir, f"vae_latent{latent_dim}_hidden{vae_hidden}.pth")

    # 2) Load or Pre-train the VAE
    if os.path.exists(vae_path) and os.path.exists(scaler_path):
        vae = ParamVAE(input_dim=d+8, latent_dim=latent_dim, hidden_dim=vae_hidden).to(device)
        vae.load_state_dict(torch.load(vae_path, map_location=device))
        sc = np.load(scaler_path)
        vae.scaler_mean.copy_(torch.tensor(sc['mean'], dtype=torch.float32).to(device))
        vae.scaler_std.copy_(torch.tensor(sc['std'],  dtype=torch.float32).to(device))
    else:
        # sample and train
        X = torch.tensor(sample_random_params(5000, d), dtype=torch.float32)
        vae = train_vae(X, latent_dim, vae_hidden, vae_epochs, vae_lr, device)
        torch.save(vae.state_dict(), vae_path)
        np.savez(scaler_path,
                 mean=vae.scaler_mean.cpu().numpy(),
                 std=vae.scaler_std.cpu().numpy())

    results_list = []
    max_performance_so_far = float('-inf')
    dataset = "_".join(evaluation_task.keys())
    run_BO_on = "vae"
    results_dir = f"{output_dir}/{dataset}/{run_BO_on}"
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"trial_{trial_number + 1}.json")
    meta_info = {
        "seed": seed,
    }
    meta_path = os.path.join(results_dir, f"trial_{trial_number + 1}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta_info, f, indent=2)
        
    for i in range(2, BO_run):
        # 3) Encode past BO inputs into latent Z
        X_bo = torch.tensor(GP_input, dtype=torch.float32).to(device)
        vae.eval()
        with torch.no_grad():
            mu, logvar = vae.encode((X_bo - vae.scaler_mean) / vae.scaler_std)
        Zp = mu.detach().cpu()
        
        sd = (0.5 * logvar).exp()
        avg_sd, z_min, z_max = sd.mean(dim=0), mu.min(0).values, mu.max(0).values
        lo = z_min - 3 * avg_sd
        hi = z_max + 3 * avg_sd
        bounds = torch.stack([lo, hi]).to(Zp.device).double()

        # 4) Fit GP on (Zp, observed_output)
        y_tensor = torch.tensor(observed_output, dtype=torch.float32).unsqueeze(-1)
        gp = SingleTaskGP(Zp.double(), y_tensor.double(), outcome_transform=Standardize(m=1))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))

        # 5) Acquire next z* with UCB
        UCB = UpperConfidenceBound(gp, beta=ucb_beta)
        z_candidate, _ = optimize_acqf(UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20)
        z_candidate = z_candidate.detach()

        # 6) Decode z* → x_candidate (18-D)
        with torch.no_grad():
            param = next(vae.parameters())
            vae_device = param.device
            vae_dtype  = param.dtype

            # move & cast the candidate z
            z_in = z_candidate.to(vae_device).to(vae_dtype)

            # now decode without a dtype mismatch
            x_candidate = vae.decode(z_in).cpu().squeeze(0).numpy()

        # 7) Discretize + renormalize
        # a) mixing dims
        mix = np.clip(x_candidate[:d], 0, None)
        if mix.sum() > 0:
            mix /= mix.sum()
        # b) num_layers
        lora_max_layers = 32
        num_layers = int(np.clip(round(x_candidate[d]), 1, lora_max_layers))
        # c) flags
        flags = (x_candidate[d+1:d+6] > 0.5).astype(float)
        # d) rank
        rank_opts = np.array([8,16,32,64,128])
        raw_rank = x_candidate[d+6]
        rank = float(rank_opts[np.argmin(np.abs(rank_opts - raw_rank))])
        # e) dropout
        max_dropout = 0.2
        dropout = float(np.clip(x_candidate[d+7], 0.0, max_dropout))

        input_X = np.concatenate([
            mix,
            [num_layers],
            flags,
            [rank],
            [dropout]
        ]).tolist()

        # if no modules are active, force at least one
        if sum(flags) == 0:
            flags[0] = 1.0
            input_X = np.concatenate([
                mix,
                [num_layers],
                flags,
                [rank],
                [dropout]
            ]).tolist()

        if printout:
            print(f"[VAE-BO] iter {i}, proposed parameters (discrete): {input_X}")

        # 8) Train & evaluate exactly as in joint_opt_BO_LLM
        tokenizer, base_model = get_tokenizer_and_model(model_id=model_id)
        base_model = base_model.to(evaluation_cuda)
        for mod in base_model.modules():
            if hasattr(mod, "weight") and isinstance(mod.weight, torch.Tensor):
                mod.dtype  = mod.weight.dtype
                mod.device = mod.weight.device
        lora_cfg = arrange_lora_config(
            input_X[-2],           # lora_rank
            input_X[-1],           # dropout
            int(input_X[d]),       # num_layers
            input_X[d+1:d+6],      # flags
        )
        if lora_cfg is None:
            # penalize completely invalid config instead of crashing
            perf = -1e6
        else:
            path_to_model = extract_data_mixture_and_train(
                model=base_model,
                random_dir=random_dir,
                tokenizer=tokenizer,
                train_datasets =[load_data(dom)[0] for dom in data_domains],
                val_datasets   =[load_data(dom)[1] for dom in data_domains],
                data_domains = data_domains,
                mixing_ratio = input_X[:d],
                additional_info=[None]*d,
                total_number_datapoints=total_data,
                run_name=f"VAE_BO_run_{i}",
                method=sampling_method,
                train_epochs=train_epochs,
                batch_size=training_batch,
                max_step=max_steps,
                lora_config=lora_cfg,
                eval_steps=eval_steps,
                callback=[time_callback],
            )
            with torch.no_grad():
                torch.cuda.empty_cache()

            peft_conf = PeftConfig.from_pretrained(path_to_model)
            model_lm = AutoModelForCausalLM.from_pretrained(peft_conf.base_model_name_or_path, torch_dtype="auto")
            lora_model = PeftModel.from_pretrained(model_lm, path_to_model).to(evaluation_cuda)
            tok = AutoTokenizer.from_pretrained(peft_conf.base_model_name_or_path, trust_remote_code=True)

            perf = 0.0
            results = evaluate_tasks(list(evaluation_task), lora_model, tok,
                                     batch=evaluation_batch, few_shot=1, limit=limit)
            for task, (w, metric) in evaluation_task.items():
                if task == "ai2_arc":
                    score = results["results"]["arc_challenge"][metric]
                else:
                    score = results["results"][task][metric]
                perf += w * ( -score if task=="wikitext" else score )

            lora_model.to("cpu")
            shutil.rmtree(path_to_model, ignore_errors=True)
            
            # Save results for this iteration
            max_performance_so_far = max(max_performance_so_far, perf)
            results_list.append({
                "iteration": i + 1,
                "mixing_ratio": [to_serializable(x) for x in input_X[:len(data_domains)]], # mixing ratio is the first len(data_domains) elements
                "model_params": [to_serializable(x) for x in input_X[len(data_domains):]], # model params are the last 6 elements
                "performance": perf,
                "max_performance_so_far": max_performance_so_far
            })
            # Write to JSON after each iteration for safety
            with open(results_path, "w") as f:
                json.dump(results_list, f, indent=2)

        if printout:
            print(f"[VAE-BO] iter {i}, performance: {perf:.6f}")

        # 9) Append for next round
        GP_input.append(input_X)
        observed_output.append(perf)

    return GP_input, observed_output, gp