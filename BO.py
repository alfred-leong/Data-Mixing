from botorch.models import SingleTaskGP
from botorch.acquisition import UpperConfidenceBound, LogExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from botorch.models.transforms.outcome import Standardize
import shutil
import torch
import os
from typing import List, Tuple, Optional
import json, hashlib, random
from itertools import product, cycle

_orig_empty = torch.empty

def _patched_empty(*args, **kwargs):
    # 1) Extract the size tuple
    if len(args) >= 1 and isinstance(args[0], (list, tuple)):
        raw_size = args[0]
        # coerce each dimension to int
        size = tuple(int(x) for x in raw_size)
        rest = args[1:]
    else:
        size = args
        rest = ()
    # 2) Drop dtype=None / device=None
    if kwargs.get("dtype", None) is None:
        kwargs.pop("dtype", None)
    if kwargs.get("device", None) is None:
        kwargs.pop("device", None)
    return _orig_empty(size, *rest, **kwargs)

# Override globally
torch.empty = _patched_empty

from itertools import product
import numpy as np
import random
from typing import Optional, List

from helper import get_data_from_mixing_ratio
from image_training import train
from typing import List
from torch.utils.data import DataLoader, TensorDataset
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

from transformers import set_seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
set_seed(42)

lora_alpha = 16
lora_dropout= 0.05
lora_r=16
lora_target_modules = [
    "q_proj",
    "v_proj",
]
lora_config = LoraConfig(
    r=int(lora_r),
    lora_alpha=lora_alpha,
    target_modules=lora_target_modules,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)
    
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
        raw = candidate[0].cpu().numpy()  # say it’s an np.ndarray of shape (d,)
        raw[raw < 0.05] = 0.0
        if raw.sum() == 0:
            raw[:] = 1.0 / len(raw)
        else:
            raw /= raw.sum()
        input_X = raw.tolist()
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
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from tqdm import tqdm

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

def arrange_lora_config(lora_r, lora_dropout, num_layers_to_apply, five_dim_vector):
    '''
    lora_r: float
    lora_dropout = float 
    num_layers_to_apply = int
    five_dim_vector = List[float]. Five dimension
    '''
    print("arranging lora config with parameters: ", lora_r, lora_dropout, num_layers_to_apply, five_dim_vector)
    
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
    lora_alpha=16,
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

def joint_opt_BO_LLM_only_data(default_rank, default_layer, default_num_layers_to_apply, default_dropout, default_alpha, time_callback, data_domains : List[str], random_dir : str, BO_run : int, total_data : int, evaluation_cuda : str, evaluation_task : dict, ucb_beta, sampling_method = "random", train_epochs : int = 1, training_batch : int = 8, evaluation_batch : int = 4, printout=True, max_steps = -1, eval_steps=100, limit=100, model_id = "LLM/llama_8b_instruct"):
    
    train_datasets = []
    val_datasets = []
    for data_domain in data_domains:
        train_dataset, val_dataset = load_data(data_domain=data_domain)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    # get tokenizer and model
    # model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    # tokenizer, model = get_tokenizer_and_model(model_id = model_id)
    
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
    
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    for i in tqdm(range(BO_run)):
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        tokenizer, model = get_tokenizer_and_model(model_id = model_id)

        print("iteration: ", i)
        print("input_X: ", input_X) 
        
        lora_config = arrange_lora_config(default_rank, default_dropout, default_num_layers_to_apply, default_layer)
        
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
        candidate, acq_value = optimize_acqf(
            UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=10,
            equality_constraints = [(torch.tensor(x), torch.tensor(A), 1)] # edit this TODO.
        )
        
        print("proposed candidate before processing:", candidate[0])
        input_X_between_0_1 = list(candidate[0])
        input_X = list(candidate[0])
        
    return GP_input, observed_output, gp

def joint_opt_BO_LLM(time_callback, lora_rank_max, data_domains : List[str], random_dir : str, BO_run : int, total_data : int, evaluation_cuda : str, evaluation_task : dict, ucb_beta, sampling_method = "random", train_epochs : int = 1, training_batch : int = 8, evaluation_batch : int = 4, printout=True, max_steps = -1, eval_steps=100, limit=100, model_id = "LLM/llama_8b_instruct"):

    train_datasets = []
    val_datasets = []
    for data_domain in data_domains:
        train_dataset, val_dataset = load_data(data_domain=data_domain)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    # get tokenizer and model
    # model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    # tokenizer, model = get_tokenizer_and_model(model_id = model_id)
    lora_max_num_layers = 32
    
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
    
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # for each BO iteration, do this...
    for i in tqdm(range(BO_run)):
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        tokenizer, model = get_tokenizer_and_model(model_id = model_id)

        print("iteration: ", i)
        print("input_X: ", input_X)
        if printout:
            print("mixing data with method: ", sampling_method)
        
        # take the model related inputs and arrange them in a nice lora config file
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
            base_path = lora_path.rsplit('/', 1)[0] + '/'
            shutil.rmtree(base_path, ignore_errors=True)
            for task in evaluation_task:
                task_weight, metric = evaluation_task[task]
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
        
        # these are updated with the candidates and used in next iteration
        input_X_between_0_1 = list(candidate[0])
        input_X = process_values(candidate[0], len(data_domains))
        
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

def joint_opt_BO_LLM_only_model(time_callback, lora_rank_max, data_domains : List[str], random_dir : str, BO_run : int, total_data : int, evaluation_cuda : str, evaluation_task : dict, ucb_beta, sampling_method = "random", train_epochs : int = 1, training_batch : int = 8, evaluation_batch : int = 4, printout=True, max_steps = -1, eval_steps=100, limit=100, model_id = "LLM/llama_8b_instruct"):
    
    train_datasets = []
    val_datasets = []
    for data_domain in data_domains:
        train_dataset, val_dataset = load_data(data_domain=data_domain)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    # get tokenizer and model
    # model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    # tokenizer, model = get_tokenizer_and_model(model_id = model_id)
    lora_max_num_layers = 32

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
        #all_influences.append(torch.load("influence/"+str(train_domain)+"_training.pt"))

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)


    for i in tqdm(range(BO_run)):
        # get tokenizer and model
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        tokenizer, model = get_tokenizer_and_model(model_id = model_id)

        print("iteration: ", i)
        print("input_X: ", input_X)
        if printout:
            print("mixing data with method: ", sampling_method)
        # lora_r, lora_dropout, num_layers_to_apply, five_dim_vector
        lora_config = arrange_lora_config(input_X[6], input_X[7], input_X[0], input_X[1:6])

        if lora_config is None:
            observed_performance = 0.1 # very bad performance if we use this
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
            base_path = lora_path.rsplit('/', 1)[0] + '/'
            shutil.rmtree(base_path, ignore_errors=True)
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

def evaluate_single_configuration(time_callback, lora_rank_max, data_domains : List[str], random_dir : str, total_data : int, evaluation_cuda : str, evaluation_task : dict, init_mixing_ratio: List[float] = None, init_lora_num_layers: int = None, init_lora_modules: List[int] = None, init_lora_rank: int = None, init_lora_dropout: float = None, sampling_method = "random", train_epochs : int = 1, training_batch : int = 8, evaluation_batch : int = 4, max_steps = -1, eval_steps=100, limit=100, model_id = "LLM/llama_8b_instruct") -> float:

    def initialise_values(init_mixing_ratio, init_lora_num_layers, init_lora_rank, init_lora_dropout):
        if init_mixing_ratio is None:
            input_X = (len(data_domains))*[float(1/len(data_domains))]
            input_X_between_0_1 = (len(data_domains))*[float(1/len(data_domains))]
        else:
            input_X = init_mixing_ratio
            input_X_between_0_1 = init_mixing_ratio
        if init_lora_num_layers is None:
            input_X.append(int(len(data_domains)*0.5))
            input_X_between_0_1.append(0.5)
        else:
            input_X.append(init_lora_num_layers)
            input_X_between_0_1.append(init_lora_num_layers/len(data_domains))
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
    
    input_X, input_X_between_0_1 = initialise_values(
        init_mixing_ratio, 
        init_lora_num_layers, 
        init_lora_rank, 
        init_lora_dropout
    )

    print("initial input_X: ", input_X)

    train_datasets = []
    val_datasets = []
    for data_domain in data_domains:
        train_dataset, val_dataset = load_data(data_domain=data_domain)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    # get tokenizer and model
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer, model = get_tokenizer_and_model(model_id = model_id)
    print("number of model layers = ", len(model.model.layers))
    print(f"Base model param count: {sum(p.numel() for p in model.parameters())}")
    
    # Create LoRA config
    lora_config = arrange_lora_config(
        input_X[-2],                         # lora rank
        input_X[-1],                         # dropout
        input_X[len(data_domains)],         # lora layer
        input_X[len(data_domains)+1:len(data_domains)+6]  # other lora params
    )

    print("LoRA config applied:", lora_config)

    all_influences = [] # not used currently
    for train_domain in data_domains:
        all_influences.append(None)

    if lora_config is None:
        return 0.1  # penalise bad config

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Train model
    path_to_final_model = extract_data_mixture_and_train(
        model=model,
        random_dir=random_dir,
        tokenizer=tokenizer,
        train_datasets=train_datasets,
        val_datasets=val_datasets,
        data_domains=data_domains,
        mixing_ratio=input_X[:len(data_domains)],
        additional_info=all_influences,
        total_number_datapoints=total_data,
        run_name="manual_eval",
        method=sampling_method,
        train_epochs=train_epochs,
        batch_size=training_batch,
        max_step=max_steps,
        lora_config=lora_config,
        eval_steps=eval_steps,
        callback=[time_callback] if time_callback else []
    )

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
    lora_model.eval()
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
    print("Observed performance: ", observed_performance)

    return observed_performance
RANKS = np.array([8, 16, 32, 64, 128], dtype=np.int32)

# Intialise with bad mixing ratios
def get_mixing_ratio(evaluation_task):
    dataset = "_".join(evaluation_task.keys())
    if dataset == "gsm8k":
        return [0,0,0.14,0.31,0.12,0.14,0,0.29,0,0]
    elif dataset == "commonsense_qa":
        return [0,0,0,1,0,0,0,0,0,0]
    elif dataset == "headqa_en":
        return [0.1221754401922226,0.0,0.539222776889801,0.0,0.0,0.0,0.2574373185634613,0.0,0.0,0.0811644196510315]
    elif dataset == "pubmedqa":
        return [0.0,0.0,0.0,0.0,0.0,0.06087161973118782,0.9391283392906189,0.0,0.0,0.0]
    elif dataset == "triviaqa":
        return [0.0,0.0,0.0,0.6801438927650452,0.11240127682685852,0.0,0.2074548453092575,0.0,0.0,0.0]
    elif dataset == "truthfulqa_gen":
        return [0.0,0.0,0.0,0.0,0.0,0.20507599413394928,0.0,0.0,0.0,0.7949240207672119]
    else:   # fallback for wikitext, mmlu, ai2_arc
        return [1,0,0,0,0,0,0,0,0,0]
# # ----------------------------
# # Sampling / domain utilities
# # ----------------------------
# def sample_random_params(num_points: int, d: int) -> np.ndarray:
#     """Draw from the SAME domain you’ll evaluate: rank∈RANKS, dropout∈[0,0.2]."""
#     out = []
#     for _ in range(num_points):
#         mix = np.random.dirichlet(np.ones(d))
#         num_layers = np.random.randint(1, 33)
#         flags = np.random.randint(0, 2, size=5)
#         rank = np.random.choice(RANKS)
#         dropout = np.random.rand() * 0.2
#         out.append(np.concatenate([mix, [num_layers], flags, [rank], [dropout]]))
#     return np.stack(out).astype(np.float32)  # (N, 18)


# def to_vae_space(x: np.ndarray, d: int) -> np.ndarray:
#     """Map original params → [0,1]-ish space for the VAE."""
#     y = x.astype(np.float32).copy()
#     # num_layers -> [0,1]
#     y[d] = (y[d] - 1.0) / 31.0
#     # flags unchanged (0/1)
#     # rank -> index in [0..4] -> [0,1]
#     idx = int(np.argmin(np.abs(RANKS - y[d + 6])))
#     y[d + 6] = idx / float(len(RANKS) - 1)
#     # dropout -> [0,1] via /0.2
#     y[d + 7] = y[d + 7] / 0.2
#     # mix already in [0,1] with sum=1 (we’ll re-softmax after decode anyway)
#     return y


# def from_vae_space(y: np.ndarray, d: int) -> np.ndarray:
#     """Map VAE output back to original domain (before robust projection)."""
#     x = y.astype(np.float32).copy()
#     x[d] = np.clip(np.round(x[d] * 31.0 + 1.0), 1, 32)  # int 1..32
#     # flags thresholded later
#     idx = int(np.clip(np.round(y[d + 6] * (len(RANKS) - 1)), 0, len(RANKS) - 1))
#     x[d + 6] = float(RANKS[idx])                         # one of allowed ranks
#     x[d + 7] = np.clip(y[d + 7] * 0.2, 0.0, 0.2)        # [0,0.2]
#     return x


# def robust_project(x_pred: np.ndarray, d: int, tau: float = 0.8) -> List[float]:
#     """Smoothly project decoded vector to a valid config."""
#     x = x_pred.astype(np.float32).copy()

#     # a) mixing -> simplex via softmax
#     mix_logits = x[:d]
#     mix_logits = mix_logits - mix_logits.max()  # stabilize
#     mix = np.exp(mix_logits / max(1e-6, tau))
#     mix = mix / (mix.sum() + 1e-8)

#     # b) num_layers already [1..32] from from_vae_space
#     num_layers = int(np.clip(np.round(x[d]), 1, 32))

#     # c) flags: ensure at least one
#     flags = (x[d + 1:d + 6] >= 0.5).astype(float)
#     if flags.sum() == 0:
#         flags[0] = 1.0

#     # d) rank already snapped
#     rank = float(x[d + 6])

#     # e) dropout in [0,0.2]
#     dropout = float(np.clip(x[d + 7], 0.0, 0.2))

#     return np.concatenate([mix, [num_layers], flags, [rank], [dropout]]).tolist()


# # ----------------------------
# # VAE
# # ----------------------------
# class ParamVAE(nn.Module):
#     def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 128):
#         super().__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc21 = nn.Linear(hidden_dim, latent_dim)
#         self.fc22 = nn.Linear(hidden_dim, latent_dim)
#         self.fc3 = nn.Linear(latent_dim, hidden_dim)
#         self.fc4 = nn.Linear(hidden_dim, input_dim)

#     def encode(self, x):
#         h = F.relu(self.fc1(x))
#         return self.fc21(h), self.fc22(h)  # mu, logvar

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         return mu + torch.randn_like(std) * std

#     def decode(self, z):
#         return self.fc4(F.relu(self.fc3(z)))

#     def forward(self, x):
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         x_hat = self.decode(z)
#         return x_hat, mu, logvar


# def train_vae(
#     x_tensor: torch.Tensor,
#     latent_dim: int,
#     hidden_dim: int,
#     epochs: int,
#     lr: float,
#     device: torch.device,
#     beta_end: float = 0.3,
#     warmup: int = 10,
# ) -> ParamVAE:
#     """Train VAE on y ∈ [0,1]-ish (outputs of to_vae_space). Light KL to avoid collapse."""
#     input_dim = x_tensor.size(1)
#     vae = ParamVAE(input_dim, latent_dim, hidden_dim).to(device)
#     opt = optim.Adam(vae.parameters(), lr=lr)

#     x = x_tensor.to(device)
#     for ep in range(epochs):
#         x_hat, mu, logvar = vae(x)
#         recon = F.mse_loss(x_hat, x)
#         kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
#         beta = beta_end if ep >= warmup else (beta_end * ep / max(1, warmup))
#         loss = recon + beta * kld

#         opt.zero_grad()
#         loss.backward()
#         opt.step()
#     return vae


# # ----------------------------
# # LoRA config (defensive)
# # ----------------------------
# def arrange_lora_config(lora_r, lora_dropout, num_layers_to_apply, five_dim_vector):
#     from peft import LoraConfig  # ensure available in your env

#     lora_r = int(max(1, round(float(lora_r))))
#     if lora_r < 8:
#         lora_r = 8
#     lora_dropout = float(lora_dropout)
#     num_layers_to_apply = int(num_layers_to_apply)

#     lora_modules_all = ["q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"]
#     flags = [int(round(float(f))) for f in five_dim_vector]
#     lora_module_to_tune = [m for m, f in zip(lora_modules_all, flags) if f == 1]

#     if not lora_module_to_tune:
#         return None

#     lora_specific_modules = []
#     for module in lora_module_to_tune:
#         if module in {"q_proj", "v_proj"}:
#             for i in range(num_layers_to_apply):
#                 lora_specific_modules.append(f"model.layers.{i}.self_attn.{module}")
#         else:
#             for i in range(num_layers_to_apply):
#                 lora_specific_modules.append(f"model.layers.{i}.mlp.{module}")

#     return LoraConfig(
#         r=lora_r,
#         lora_alpha=16,
#         target_modules=lora_specific_modules,
#         lora_dropout=lora_dropout,
#         bias="none",
#         task_type="CAUSAL_LM",
#     )


# # ----------------------------
# # BO with VAE (unit-cube + TR)
# # ----------------------------
# def joint_opt_BO_LLM_with_vae(
#     time_callback,
#     lora_rank_max: int,
#     data_domains: List[str],
#     random_dir: str,
#     BO_run: int,
#     total_data: int,
#     evaluation_cuda: str,
#     evaluation_task: dict,
#     ucb_beta: float,
#     sampling_method="random",
#     train_epochs: int = 1,
#     training_batch: int = 8,
#     evaluation_batch: int = 4,
#     printout=True,
#     max_steps: int = -1,
#     eval_steps: int = 100,
#     limit=100,
#     model_id="meta-llama/Meta-Llama-3-8B-Instruct",
#     latent_dim: int = 10,
#     vae_hidden: int = 128,
#     vae_epochs: int = 50,
#     vae_lr: float = 1e-3,
# ):
#     # 1) Warm start with standard BO for 2 steps (your existing function)
#     warmup_X, warmup_y, _ = joint_opt_BO_LLM(
#         time_callback, lora_rank_max, data_domains, random_dir,
#         2, total_data, evaluation_cuda, evaluation_task,
#         ucb_beta, sampling_method,
#         train_epochs, training_batch, evaluation_batch,
#         printout, max_steps, eval_steps, limit, model_id,
#     )
#     GP_input = [list(x) for x in warmup_X]  # list of 18-d configs
#     observed_output = list(warmup_y)

#     d = len(data_domains)
#     device = torch.device(
#         evaluation_cuda if (isinstance(evaluation_cuda, str) and evaluation_cuda.startswith("cuda"))
#         else (f"cuda:{int(evaluation_cuda)}" if torch.cuda.is_available() else "cpu")
#     )
#     # 2) Load or train VAE on normalized params
#     vae_dir = os.path.join(os.getcwd(), "vae")
#     os.makedirs(vae_dir, exist_ok=True)
#     vae_path = os.path.join(vae_dir, f"vae_ld{latent_dim}_hd{vae_hidden}.pth")

#     if os.path.exists(vae_path):
#         vae = ParamVAE(d + 8, latent_dim, vae_hidden).to(device)
#         vae.load_state_dict(torch.load(vae_path, map_location=device))
#     else:
#         X = sample_random_params(5000, d)                   # original domain
#         Y = np.vstack([to_vae_space(x, d) for x in X])      # normalized
#         vae = train_vae(torch.tensor(Y, dtype=torch.float32), latent_dim, vae_hidden,
#                         epochs=vae_epochs, lr=vae_lr, device=device)
#         torch.save(vae.state_dict(), vae_path)

#     # Trust-region radius (in unit space)
#     tr_r = 0.3
#     no_improve = 0
#     best_so_far = max(observed_output)

#     for i in range(2, BO_run):
#         # 3) Encode past X into latent means μ (deterministic)
#         X_bo = np.vstack([to_vae_space(np.array(x, dtype=np.float32), d) for x in GP_input])
#         X_bo_t = torch.tensor(X_bo, dtype=torch.float32, device=device)
#         vae.eval()
#         with torch.no_grad():
#             mu, _ = vae.encode(X_bo_t)
#         Zp = mu.detach().cpu()  # (n, latent_dim), float32 on CPU

#         # 4) Fit GP on unit cube
#         y_tensor = torch.tensor(observed_output, dtype=torch.float32).unsqueeze(-1)
#         Zmin, Zmax = Zp.min(0).values, Zp.max(0).values
#         Zrng = (Zmax - Zmin).clamp_min(1e-8)
#         Z_unit = (Zp - Zmin) / Zrng

#         gp = SingleTaskGP(Z_unit.double(), y_tensor.double(), outcome_transform=Standardize(m=1))
#         fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))

#         # 5) Trust region bounds around current best (in unit space)
#         best_idx = int(np.argmax(observed_output))
#         center = Z_unit[best_idx]
#         lo_uc = (center - tr_r).clamp(0.0, 1.0)
#         hi_uc = (center + tr_r).clamp(0.0, 1.0)
#         bounds = torch.stack([lo_uc.double(), hi_uc.double()])  # (2, d) double

#         # 6) Acquire next z* in unit space and map back
#         UCB = UpperConfidenceBound(gp, beta=ucb_beta)  # start with 2–3 typically
#         z_uc, _ = optimize_acqf(UCB, bounds=bounds, q=1, num_restarts=20, raw_samples=256)
#         z_uc = z_uc.squeeze(0).cpu()
#         z_candidate = (Zmin + z_uc * Zrng)  # back to latent coords (CPU)

#         # 7) Decode → normalized ŷ → back to original domain → robust projection
#         with torch.no_grad():
#             param = next(vae.parameters())
#             z_in = z_candidate.to(param.device).to(param.dtype).unsqueeze(0)
#             y_hat = vae.decode(z_in).cpu().squeeze(0).numpy()

#         # Clamp normalized outputs a bit before inverse map
#         y_hat = np.clip(y_hat, 0.0, 1.0)
#         x_back = from_vae_space(y_hat, d)                   # back to original ranges
#         input_X = robust_project(x_back, d, tau=0.8)        # final valid config

#         if printout:
#             print(f"[VAE-BO] iter {i}, proposed parameters:", input_X)

#         # 8) Train & evaluate (your existing pipeline)
#         tokenizer, base_model = get_tokenizer_and_model(model_id=model_id)
#         base_model = base_model.to(device)
#         for mod in base_model.modules():
#             if hasattr(mod, "weight") and isinstance(mod.weight, torch.Tensor):
#                 mod.dtype = mod.weight.dtype
#                 mod.device = mod.weight.device

#         lora_cfg = arrange_lora_config(
#             lora_r=input_X[-2],              # rank
#             lora_dropout=input_X[-1],        # dropout
#             num_layers_to_apply=int(input_X[d]),    # num_layers
#             five_dim_vector=input_X[d + 1:d + 6],   # flags
#         )
#         if lora_cfg is None:
#             # Repair to a safe default instead of tanking the run
#             lora_cfg = arrange_lora_config(16, 0.05, min(int(input_X[d]), 8), [1, 0, 1, 0, 0])

#         path_to_model = extract_data_mixture_and_train(
#             model=base_model,
#             random_dir=random_dir,
#             tokenizer=tokenizer,
#             train_datasets=[load_data(dom)[0] for dom in data_domains],
#             val_datasets=[load_data(dom)[1] for dom in data_domains],
#             data_domains=data_domains,
#             mixing_ratio=input_X[:d],
#             additional_info=[None] * d,
#             total_number_datapoints=total_data,
#             run_name=f"VAE_BO_run_{i}",
#             method=sampling_method,
#             train_epochs=train_epochs,
#             batch_size=training_batch,
#             max_step=max_steps,
#             lora_config=lora_cfg,
#             eval_steps=eval_steps,
#             callback=[time_callback],
#         )
#         with torch.no_grad():
#             torch.cuda.empty_cache()

#         from peft import PeftConfig, PeftModel
#         from transformers import AutoModelForCausalLM, AutoTokenizer

#         peft_conf = PeftConfig.from_pretrained(path_to_model)
#         model_lm = AutoModelForCausalLM.from_pretrained(peft_conf.base_model_name_or_path, torch_dtype="auto")
#         lora_model = PeftModel.from_pretrained(model_lm, path_to_model).to(device)
#         tok = AutoTokenizer.from_pretrained(peft_conf.base_model_name_or_path, trust_remote_code=True)

#         perf = 0.0
#         results = evaluate_tasks(list(evaluation_task), lora_model, tok,
#                                  batch=evaluation_batch, few_shot=1, limit=limit)
#         for task, (w, metric) in evaluation_task.items():
#             score = results["results"][task][metric]
#             perf += w * (-score if task == "wikitext" else score)

#         lora_model.to("cpu")
#         shutil.rmtree(path_to_model, ignore_errors=True)

#         if printout:
#             print(f"[VAE-BO] iter {i}, performance: {perf:.6f}")

#         # 9) Append & adapt TR radius
#         GP_input.append(input_X)
#         observed_output.append(perf)

#         if perf > best_so_far + 1e-9:
#             best_so_far = perf
#             tr_r = min(0.5, tr_r * 1.1)  # expand slightly
#             no_improve = 0
#         else:
#             no_improve += 1
#             if no_improve >= 5:
#                 tr_r = max(0.15, tr_r * 0.8)  # shrink if stuck
#                 no_improve = 0

#     return GP_input, observed_output, gp

# =========================
# Utilities / helpers
# =========================
# import copy, time, re

# def _safe_slug(s: str) -> str:
#     return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(s))[:64] or "x"

# def _task_name_from(evaluation_task) -> str:
#     # Accept either a single string or your dict {task: (w, metric)}
#     if isinstance(evaluation_task, str):
#         return evaluation_task
#     if isinstance(evaluation_task, dict) and len(evaluation_task) > 0:
#         return next(iter(evaluation_task.keys()))
#     return "unknown_task"

# def _uniform_flag_patterns(n: int, k: int = 5, seed: int = 42) -> List[List[float]]:
#     """
#     Returns n flag vectors of length k, drawn by cycling through all 2^k patterns
#     in a shuffled order so every pattern appears roughly equally often.
#     """
#     pats = [p for p in product([0.0, 1.0], repeat=k) if any(p)]  # drop (0,0,0,0,0)
#     rnd = random.Random(seed)
#     rnd.shuffle(pats)
#     cyc = cycle(pats)
#     return [list(next(cyc)) for _ in range(n)]

# def _renorm_simplex(vec: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
#     s = vec.sum()
#     if s <= eps:
#         return torch.full_like(vec, 1.0 / vec.numel())
#     return vec / s

# def _threshold_and_renorm_simplex(vec: torch.Tensor, floor: float = 0.01) -> torch.Tensor:
#     v = torch.clamp(vec, min=floor)
#     return v / (v.sum() + 1e-9)

# def _to_device(t, device):
#     return t.to(device) if torch.is_tensor(t) else t

# def _short_hash(d: dict) -> str:
#     # stable tiny hash for filenames
#     s = json.dumps(d, sort_keys=True, separators=(",", ":"))
#     return hashlib.sha1(s.encode()).hexdigest()[:8]

# # =========================
# # VAE (same as before, refactored)
# # =========================

# class ConfigVAE(nn.Module):
#     def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 64, mixture_dim: int = 2):
#         super().__init__()
#         self.input_dim = input_dim
#         self.latent_dim = latent_dim
#         self.mixture_dim = mixture_dim
#         cont_dim = input_dim - mixture_dim

#         self.enc = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#         )
#         self.mu = nn.Linear(hidden_dim, latent_dim)
#         self.logvar = nn.Linear(hidden_dim, latent_dim)

#         self.dec = nn.Sequential(
#             nn.Linear(latent_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#         )
#         self.dec_mix = nn.Linear(hidden_dim, mixture_dim)
#         self.dec_rest = nn.Linear(hidden_dim, cont_dim)

#     def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         h = self.enc(x)
#         return self.mu(h), self.logvar(h)

#     def reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         h = self.dec(z)
#         mix_logits = self.dec_mix(h)
#         mix = F.softmax(mix_logits, dim=-1)  # simplex
#         rest = self.dec_rest(h)              # squashed externally
#         return mix, rest

#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#         mu, logvar = self.encode(x)
#         z = self.reparam(mu, logvar)
#         mix, rest = self.decode(z)
#         return mix, rest, mu, logvar

# def vae_loss(mix_pred, rest_pred, x_target, mixture_dim, beta_kl=3.0, mu=None, logvar=None):
#     x_mix = x_target[:, :mixture_dim]
#     x_rest = x_target[:, mixture_dim:]  # flags 0/1, layers01 [0,1], rank01 [0,1], dropout [0,0.1]

#     rest_sig = torch.sigmoid(rest_pred)     # [0,1]
#     rest_sig_scaled = rest_sig.clone()
#     rest_sig_scaled[:, -1] = rest_sig_scaled[:, -1] * 0.1  # scale dropout to [0,0.1]

#     recon_mix  = F.mse_loss(mix_pred, x_mix, reduction='mean')
#     recon_rest = F.mse_loss(rest_sig_scaled, x_rest, reduction='mean')
#     recon = recon_mix + recon_rest

#     kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
#     return recon + beta_kl * kl, recon_mix.item(), recon_rest.item(), kl.item()

# def train_vae(x_train: torch.Tensor, latent_dim: int, hidden_dim: int, epochs: int, lr: float, mixture_dim: int, batch_size: int = 64, device: str = "cpu"):
#     model = ConfigVAE(input_dim=x_train.shape[1], latent_dim=latent_dim, hidden_dim=hidden_dim, mixture_dim=mixture_dim).to(device)
#     opt = torch.optim.Adam(model.parameters(), lr=lr)
#     ds = TensorDataset(x_train)
#     dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

#     model.train()
#     for _ in range(epochs):
#         for (xb,) in dl:
#             xb = _to_device(xb, device)
#             mix, rest, mu, logvar = model(xb)
#             loss, _, _, _ = vae_loss(mix, rest, xb, mixture_dim, beta_kl=3.0, mu=mu, logvar=logvar)
#             opt.zero_grad(set_to_none=True)
#             loss.backward()
#             opt.step()
#     model.eval()
#     return model

# @torch.no_grad()
# def vae_encode(model: ConfigVAE, x: torch.Tensor) -> torch.Tensor:
#     device = next(model.parameters()).device
#     mu, logvar = model.encode(x.to(device))
#     return mu.detach().cpu()   # ensure CPU for BoTorch/GP

# @torch.no_grad()
# def vae_decode_to_config(model: ConfigVAE,
#                          z: torch.Tensor,
#                          len_domains: int,
#                          lora_max_layers: int,
#                          lora_rank_max: int) -> Tuple[List[float], List[float]]:
#     device = next(model.parameters()).device
#     mix_soft, rest_lin = model.decode(z.to(device).unsqueeze(0))
#     mix_soft = mix_soft.squeeze(0).detach().cpu()           # simplex (soft)
#     rest_sig = torch.sigmoid(rest_lin.squeeze(0)).detach().cpu()

#     layers01 = rest_sig[0]
#     flags01  = rest_sig[1:6]
#     rank01   = rest_sig[6]
#     drop01   = rest_sig[7]

#     mix_exec   = _threshold_and_renorm_simplex(mix_soft, floor=0.01).tolist()
#     num_layers = int(round(lora_max_layers * layers01.item()))
#     num_layers = max(0, min(lora_max_layers, num_layers))

#     flags_disc = [int(round(v.item())) for v in flags01]    # DISCRETE for execution
#     if sum(flags_disc) == 0:
#         j = int(torch.argmax(flags01).item())
#         flags_disc[j] = 1

#     rank = int(round(lora_rank_max * rank01.item()))
#     rank = max(0, min(lora_rank_max, rank))
#     dropout = float(drop01.item() * 0.1)                    # [0,0.1]

#     input_X_real = mix_exec + [num_layers] + flags_disc + [rank] + [dropout]

#     # Important: keep mix SOFT, flags CONTINUOUS
#     x01_between_0_1 = torch.cat([
#         mix_soft,
#         layers01.view(1),
#         flags01,                 # continuous (no rounding/clamping)
#         rank01.view(1),
#         (drop01.view(1) * 0.1)   # store as [0,0.1]
#     ]).tolist()

#     return input_X_real, x01_between_0_1
# # =========================
# # Normalisation helpers
# # =========================

# def make_initial_configs(len_domains: int, lora_max_layers: int, lora_rank_max: int):
#     mix = [1.0 / len_domains] * len_domains
#     layers = int(lora_max_layers * 0.5)
#     flags = [1, 1, 1, 1, 1]
#     rank  = 72
#     dropout = 0.05
#     x01 = mix + [0.5] + [1, 1, 1, 1, 1] + [72.0/lora_rank_max] + [dropout]
#     return (mix + [layers] + flags + [rank] + [dropout]), x01

# # =========================
# # Random normalised sampling
# # =========================

# def sample_random_normalised(len_domains: int, flags01: Optional[List[float]] = None) -> List[float]:
#     # mixture via Dirichlet (simplex)
#     mix = torch.distributions.Dirichlet(torch.ones(len_domains)).sample().tolist()
#     layers01 = random.random()
#     # CHANGED: use provided flags01 if given; otherwise Bernoulli(0.5) fallback
#     if flags01 is None:
#         flags01 = [float(random.random() < 0.5) for _ in range(5)]
#     rank01   = random.random()
#     dropout  = random.random() * 0.1  # stored as 0..0.1
#     return mix + [layers01] + flags01 + [rank01] + [dropout]

# def random_dataset_normalised(n: int, len_domains: int) -> torch.Tensor:
#     flags_pool = _uniform_flag_patterns(
#         n=n,
#         k=5,
#         seed=random.randint(0, 2**31 - 1)  # different processes get different orders
#     )
#     data = []
#     for i in range(n):
#         # NOTE: pass a specific flags pattern to force uniform coverage
#         data.append(sample_random_normalised(len_domains, flags01=flags_pool[i]))
#     return torch.tensor(data, dtype=torch.float32)

# # =========================
# # Save / Load VAE
# # =========================

# def _vae_key(latent_dim: int, hidden_dim: int, mixture_dim: int, epochs: int, lr: float) -> str:
#     meta = {
#         "ld": latent_dim,
#         "hd": hidden_dim,
#         "md": mixture_dim,
#         "ep": epochs,
#         "lr": lr,
#     }
#     return f"vae_ld{latent_dim}_hd{hidden_dim}_md{mixture_dim}_{_short_hash(meta)}"

# def save_vae(model: ConfigVAE, save_dir: str, key: str):
#     os.makedirs(save_dir, exist_ok=True)
#     path = os.path.join(save_dir, f"{key}.pt")
#     torch.save({"state_dict": model.state_dict(),
#                 "input_dim": model.input_dim,
#                 "latent_dim": model.latent_dim,
#                 "hidden_dim": model.enc[0].out_features,
#                 "mixture_dim": model.mixture_dim}, path)

# def load_vae_if_exists(save_dir: str, key: str) -> Optional[ConfigVAE]:
#     path = os.path.join(save_dir, f"{key}.pt")
#     if not os.path.isfile(path):
#         return None
#     ckpt = torch.load(path, map_location="cpu")
#     model = ConfigVAE(
#         input_dim=ckpt["input_dim"],
#         latent_dim=ckpt["latent_dim"],
#         hidden_dim=ckpt["hidden_dim"],
#         mixture_dim=ckpt["mixture_dim"],
#     )
#     model.load_state_dict(ckpt["state_dict"])
#     model.eval()
#     return model

# def fine_tune_vae(model: ConfigVAE,
#                   x_train: torch.Tensor,
#                   mixture_dim: int,
#                   epochs: int = 5,
#                   lr: float = 1e-3,
#                   batch_size: int = 64,
#                   device: str = "cpu") -> ConfigVAE:
#     model = model.to(device)
#     ds = TensorDataset(x_train)
#     dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
#     opt = torch.optim.Adam(model.parameters(), lr=lr)
#     model.train()
#     for _ in range(epochs):
#         for (xb,) in dl:
#             xb = xb.to(device)
#             mix, rest, mu, logvar = model(xb)
#             loss, _, _, _ = vae_loss(mix, rest, xb, mixture_dim, beta_kl=3.0, mu=mu, logvar=logvar)
#             opt.zero_grad(set_to_none=True)
#             loss.backward()
#             opt.step()
#     model.eval()
#     return model
# # =========================
# # Main: VAE-BO
# # =========================

# def joint_opt_BO_LLM_with_vae(
#     time_callback,
#     lora_rank_max: int,
#     data_domains: List[str],
#     random_dir: str,
#     BO_run: int,
#     total_data: int,
#     evaluation_cuda: str,
#     evaluation_task: dict,
#     ucb_beta: float,
#     sampling_method: str = "random",
#     train_epochs: int = 1,
#     training_batch: int = 8,
#     evaluation_batch: int = 4,
#     printout: bool = True,
#     max_steps: int = -1,
#     eval_steps: int = 100,
#     limit: int = 100,
#     latent_dim: int = 10,
#     vae_hidden: int = 64,
#     vae_epochs: int = 20,
#     vae_lr: float = 1e-3,
#     # NEW:
#     vae_pretrain_samples: int = 2000,         # how many random configs to pretrain on
#     vae_model_dir: str = "./vae_models",     # where to save/load VAE checkpoints
#     model_id: str = "LLM/llama_8b_instruct",
# ):
#     """
#     Returns:
#         GP_input: List[List[float]]   (history of normalised configs)
#         observed_output: List[float]
#         gp: SingleTaskGP
#     """
#     # 1) Load datasets once
#     train_datasets, val_datasets = [], []
#     for dom in data_domains:
#         tr, va = load_data(data_domain=dom)
#         train_datasets.append(tr)
#         val_datasets.append(va)

#     lora_max_num_layers = 32
#     len_domains = len(data_domains)
#     vae_device = "cuda" if torch.cuda.is_available() else "cpu"

#     # 2) Initial config
#     bad_mix = get_mixing_ratio(evaluation_task)
#     if len(bad_mix) != len_domains:
#         if printout:
#             print(f"[VAE-BO] Warning: bad_mix length {len(bad_mix)} != len_domains {len_domains}. Adjusting.")
#         if len(bad_mix) > len_domains:
#             bad_mix = bad_mix[:len_domains]
#         else:
#             bad_mix = bad_mix + [0.0] * (len_domains - len(bad_mix))
#     bad_mix = (_renorm_simplex(torch.tensor(bad_mix, dtype=torch.float32))).tolist()

#     layers = int(lora_max_num_layers * 0.5)
#     flags  = [1, 1, 1, 1, 1]
#     rank   = 72
#     dropout = 0.05

#     # real + normalised vectors (dropout kept in [0,0.1])
#     input_X = bad_mix + [layers] + flags + [rank] + [dropout]
#     input_X_between_0_1 = bad_mix + [0.5] + flags + [72.0 / float(max(lora_rank_max, 1))] + [dropout]

#     GP_input: List[List[float]] = []
#     observed_output: List[float] = []

#     # # 3) Pretrain VAE on random configs (or load if existing)
#     # vae_id = _vae_key(latent_dim, vae_hidden, len_domains, vae_epochs, vae_lr)
#     # vae = load_vae_if_exists(vae_model_dir, vae_id)
#     # if vae is None:
#     #     if printout:
#     #         print(f"[VAE-BO] Pretraining VAE ({vae_id}) on {vae_pretrain_samples} random configs...")
#     #     Xpre = random_dataset_normalised(vae_pretrain_samples, len_domains)
#     #     vae = train_vae(
#     #         x_train=Xpre,
#     #         latent_dim=latent_dim,
#     #         hidden_dim=vae_hidden,
#     #         epochs=vae_epochs,
#     #         lr=vae_lr,
#     #         mixture_dim=len_domains,
#     #         batch_size=64,
#     #         device=vae_device,
#     #     )
#     #     save_vae(vae, vae_model_dir, vae_id)
#     #     if printout:
#     #         print(f"[VAE-BO] Saved VAE to {os.path.join(vae_model_dir, vae_id + '.pt')}")
#     # else:
#     #     if printout:
#     #         print(f"[VAE-BO] Loaded pretrained VAE: {vae_id}")
#     task_name = _task_name_from(evaluation_task)

#     # 19aug fix to allow parallel runs with different tasks
#     task_slug = _safe_slug(task_name)

#     vae_base_id = "BASE__" + _vae_key(latent_dim, vae_hidden, len_domains, vae_epochs, vae_lr)    # CHANGED
#     vae_task_id = f"TASK__{task_slug}__" + _vae_key(latent_dim, vae_hidden, len_domains, vae_epochs, vae_lr)  # CHANGED

#     # 3a) Ensure a single base VAE exists (task-agnostic)
#     base_vae = load_vae_if_exists(vae_model_dir, vae_base_id)
#     if base_vae is None:
#         if printout:
#             print(f"[VAE-BO] Pretraining BASE VAE ({vae_base_id}) on {vae_pretrain_samples} random configs...")
#         Xpre = random_dataset_normalised(vae_pretrain_samples, len_domains)
#         base_vae = train_vae(
#             x_train=Xpre,
#             latent_dim=latent_dim,
#             hidden_dim=vae_hidden,
#             epochs=vae_epochs,
#             lr=vae_lr,
#             mixture_dim=len_domains,
#             batch_size=64,
#             device=vae_device,
#         )
#         save_vae(base_vae, vae_model_dir, vae_base_id)
#         if printout:
#             print(f"[VAE-BO] Saved BASE VAE to {os.path.join(vae_model_dir, vae_base_id + '.pt')}")
#     else:
#         if printout:
#             print(f"[VAE-BO] Loaded BASE VAE: {vae_base_id}")

#     # 3b) Try to load a task-specific VAE; if missing, duplicate the base
#     vae = load_vae_if_exists(vae_model_dir, vae_task_id)
#     if vae is None:
#         vae = copy.deepcopy(base_vae)  # CHANGED: duplicate, then fine-tune this copy only
#         if printout:
#             print(f"[VAE-BO] No task VAE for '{task_name}'. Duplicated BASE → {vae_task_id}")
#     else:
#         if printout:
#             print(f"[VAE-BO] Loaded task-specific VAE: {vae_task_id}")

#     # Helper: evaluate a real config (train LoRA, eval, score)
#     def evaluate_config(inp_real: List[float]) -> float:
#         num_layers = int(inp_real[len_domains])
#         layer_flags = inp_real[len_domains+1:len_domains+6]
#         rank = int(inp_real[len_domains+6])
#         dropout = float(inp_real[len_domains+7])

#         lora_cfg = arrange_lora_config(rank, dropout, num_layers, layer_flags)
#         if lora_cfg is None:
#             return 0.1

#         tokenizer, base_model = get_tokenizer_and_model(model_id="meta-llama/Meta-Llama-3-8B-Instruct")
#         path = extract_data_mixture_and_train(
#             model=base_model,
#             random_dir=random_dir,
#             tokenizer=tokenizer,
#             train_datasets=train_datasets,
#             val_datasets=val_datasets,
#             data_domains=data_domains,
#             mixing_ratio=inp_real[:len_domains],
#             additional_info=[None]*len_domains,
#             total_number_datapoints=total_data,
#             run_name=f"BO_VAE_run",
#             method=sampling_method,
#             train_epochs=train_epochs,
#             batch_size=training_batch,
#             max_step=max_steps,
#             lora_config=lora_cfg,
#             eval_steps=eval_steps,
#             callback=[time_callback],
#         )
#         with torch.no_grad():
#             torch.cuda.empty_cache()

#         lora_path = os.path.abspath(path)
#         if not os.path.isdir(lora_path):
#             raise FileNotFoundError(f"LoRA adapter directory not found: {lora_path}")
#         cfg = PeftConfig.from_pretrained(lora_path, local_files_only=True)
#         base = AutoModelForCausalLM.from_pretrained(cfg.base_model_name_or_path, torch_dtype='auto')
#         lora_model = PeftModel.from_pretrained(base, lora_path, local_files_only=True).to(evaluation_cuda)
#         tok = AutoTokenizer.from_pretrained(cfg.base_model_name_or_path, trust_remote_code=True)

#         lora_model.eval()
#         tasks = list(evaluation_task.keys())
#         results = evaluate_tasks(tasks, lora_model, tok, evaluation_batch, few_shot=1, limit=limit)

#         score = 0.0
#         for task in evaluation_task:
#             w, metric = evaluation_task[task]
#             perf = results["results"][task][metric]
#             if task == "wikitext":
#                 perf = -perf
#             score += w * perf
        
#         shutil.rmtree(lora_path, ignore_errors=True)
#         lora_model.to("cpu")
#         return float(score)

#     # 4) First evaluation
#     if printout:
#         print("[VAE-BO] Iteration 0 (bad mixing ratio if available)")
#     y0 = evaluate_config(input_X)
#     GP_input.append(list(input_X_between_0_1))
#     observed_output.append(y0)
#     if printout:
#         print(f"[VAE-BO] score = {y0:.6f}")
    

#     # # 5) Small warmup of extra random points (optional but helpful)
#     # WARMUP_RANDOM = max(3, min(8, BO_run // 3))
#     # while len(GP_input) < min(WARMUP_RANDOM + 1, BO_run):
#     #     x01 = sample_random_normalised(len_domains)
#     #     mix = torch.tensor(x01[:len_domains])
#     #     mix = _threshold_and_renorm_simplex(mix, thr=0.05).tolist()
#     #     num_layers = int(round(lora_max_num_layers * x01[len_domains]))
#     #     flags = [int(round(v)) for v in x01[len_domains+1:len_domains+6]]
#     #     rank  = int(round(lora_rank_max * x01[len_domains+6]))
#     #     dropout = float(x01[len_domains+7])  # [0,0.1]
#     #     x_real = mix + [num_layers] + flags + [rank] + [dropout]
#     #     y = evaluate_config(x_real)
#     #     GP_input.append(list(x01))
#     #     observed_output.append(y)
#     #     if printout:
#     #         print(f"[VAE-BO] Warmup {len(GP_input)-1}: score={y:.6f}")
#     # 1) Structured flags warm-up
#     if BO_run > 1:
#         FLAG_WARMUP_PATTERNS = [
#             [1,0,0,0,0],
#             [0,1,0,0,0],
#             [0,0,1,0,0],
#             [0,0,0,1,0],
#             [0,0,0,0,1],      # add this one too
#             [1,0,1,0,0],      # (optional) a couple of 2-hots
#         ]
#         max_extra = max(0, min(len(FLAG_WARMUP_PATTERNS), BO_run - len(GP_input)))
#         for flags in FLAG_WARMUP_PATTERNS[:max_extra]:
#             x_real = bad_mix + [layers] + flags + [rank] + [dropout]
#             x01    = bad_mix + [0.5]   + flags + [72.0/float(max(lora_rank_max,1))] + [dropout]
#             y = evaluate_config(x_real)
#             GP_input.append(list(x01))
#             observed_output.append(y)
#             if printout:
#                 print(f"[VAE-BO] Warm-up (flags={flags}) score = {y:.4f}")

#     # # 2) Random warm-up (a few random points)
#     # WARMUP_RANDOM = min(8, max(3, BO_run // 10))
#     # while len(GP_input) < min(WARMUP_RANDOM + 1, BO_run):  # +1 for the initial point already added
#     #     x01 = sample_random_normalised(len_domains)
#     #     mix = _threshold_and_renorm_simplex(torch.tensor(x01[:len_domains]), floor=0.01).tolist()
#     #     num_layers = int(round(lora_max_num_layers * x01[len_domains]))
#     #     flags = [int(round(v)) for v in x01[len_domains+1:len_domains+6]]
#     #     rank  = int(round(lora_rank_max * x01[len_domains+6]))
#     #     dropout = float(x01[len_domains+7])  # [0,0.1]
#     #     x_real = mix + [num_layers] + flags + [rank] + [dropout]
#     #     y = evaluate_config(x_real)
#     #     # TODO: Check this
#     #     GP_input.append(list(x01))
#     #     observed_output.append(y)
#     #     if printout:
#     #         print(f"[VAE-BO] Warm-up (random) score = {y:.4f}")
#     # 6) BO iterations in latent space
#     gp = None
#     for it in range(len(GP_input), BO_run):
#         if printout:
#             print(f"[VAE-BO] Iteration {it}: fine-tune VAE on history, fit GP, optimise UCB")

#         # Fine-tune the loaded VAE a little on true history for alignment
#         X01 = torch.tensor(GP_input, dtype=torch.float32)
#         vae = fine_tune_vae(
#             vae, X01, mixture_dim=len_domains,
#             epochs=max(3, vae_epochs // 5), lr=vae_lr,
#             batch_size=64, device=vae_device
#         )

#         # TODO: Update Z to take the previous value. if it=
#         Z = GP_input
#         Y = torch.tensor(observed_output, dtype=torch.double).view(-1, 1)

#         gp = SingleTaskGP(Z, Y)
#         mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
#         fit_gpytorch_mll(mll)

#         UCB = UpperConfidenceBound(gp, beta=ucb_beta)
#         bounds = torch.stack([-3.0 * torch.ones(latent_dim, dtype=torch.double),
#                                3.0 * torch.ones(latent_dim, dtype=torch.double)])
#         candidate_z, _ = optimize_acqf(
#             UCB,
#             bounds=bounds,
#             q=1,
#             num_restarts=10,
#             raw_samples=256,
#         )
#         z_star = candidate_z[0].float()

#         x_real, x01 = vae_decode_to_config(
#             model=vae,
#             z=z_star,
#             len_domains=len_domains,
#             lora_max_layers=lora_max_num_layers,
#             lora_rank_max=lora_rank_max,
#         )

#         if printout:
#             print("[VAE-BO] Proposed (decoded) config:")
#             print("  mixture:", [round(v, 4) for v in x_real[:len_domains]])
#             print("  layers:", x_real[len_domains])
#             print("  flags :", x_real[len_domains+1:len_domains+6])
#             print("  rank  :", x_real[len_domains+6])
#             print("  drop  :", round(x_real[len_domains+7], 4))

#         # TODO: Change x01 to zstar
#         y_star = evaluate_config(x_real)
#         # GP_input.append(list(x01))
#         GP_input.append(z_star)
#         observed_output.append(y_star)
#         if printout:
#             print(f"[VAE-BO] score = {y_star:.6f}", flush=True)
        

#     # Optional: save the updated VAE (fine-tuned on history) for reuse next time
#     save_vae(vae, vae_model_dir, vae_task_id)
#     if printout:
#         print(f"[VAE-BO] Saved task-specific VAE: {vae_task_id}")

#     # Final GP fit for return
#     if len(GP_input) > 0:
#         X01 = torch.tensor(GP_input, dtype=torch.float32)
#         with torch.no_grad():
#             Z = vae_encode(vae, X01).double()
#         Y = torch.tensor(observed_output, dtype=torch.double).view(-1, 1)
#         gp = SingleTaskGP(Z, Y)
#         mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
#         fit_gpytorch_mll(mll)

#     return GP_input, observed_output, gp

RANKS = np.array([8, 16, 32, 64, 128], dtype=np.int32)

# Maps unbounded VAE latent space to [0,1], as well as the inverse mapping
class VAEToUnitMapper:
    def __init__(self, eps: float = 1e-6):
        self.loc = None   # torch.Tensor[d]
        self.scale = None 
        self.eps = eps

    @staticmethod
    def _to_tensor(x, device):
        if isinstance(x, torch.Tensor):
            return x
        return torch.tensor(x, dtype=torch.float32, device=device)

    # vae_means: (N, d) encoder means from your VAE.
    # scale_multiplier: scale multiplier on per-dim std (soft range ~ mean ± k*std)
    # lower_bound: lower bound to avoid constant dims.
  
    def fit(self, vae_means: torch.Tensor, scale_multiplier: float = 2.0, lower_bound: float = 0.5):
        assert vae_means.ndim == 2, "mu_all should be (N, d)"
        device = vae_means.device
        mean_vector = vae_means.mean(dim=0)                    
        std_deviation = vae_means.std(dim=0)                     
        std_deviation = torch.clamp(std_deviation, min=1e-8)
        scale = torch.clamp(scale_multiplier * std_deviation, min=lower_bound) 
        self.loc = mean_vector.detach()
        self.scale = scale.detach()
        return self

    # Map unbounded VAE space to [0,1] (GP input)
    def to_unit(self, input: torch.Tensor) -> torch.Tensor:
        assert self.loc is not None, "Call fit() first."
        input = input.to(self.loc.device)
        unit_output = torch.sigmoid((input - self.loc) / self.scale)
        # Clamp output to [0,1]
        return torch.clamp(unit_output, self.eps, 1.0 - self.eps)

    # Map GP output from [0,1] to latent VAE space
    def from_unit(self, unit_input: torch.Tensor) -> torch.Tensor:
        assert self.loc is not None, "Call fit() first."
        unit_input = unit_input.to(self.loc.device)
        unit_input = torch.clamp(unit_input, self.eps, 1.0 - self.eps)
        scaled_output = self.loc + self.scale * torch.log(unit_input / (1.0 - unit_input))
        return scaled_output

def get_mixing_ratio(evaluation_task):
    dataset = "_".join(evaluation_task.keys())
    if dataset == "gsm8k":
        return [0,0,0.14,0.31,0.12,0.14,0,0.29,0,0]
    elif dataset == "commonsense_qa":
        return [0,0,0,1,0,0,0,0,0,0]
    elif dataset == "headqa_en":
        return [0.1221754401922226,0.0,0.539222776889801,0.0,0.0,0.0,0.2574373185634613,0.0,0.0,0.0811644196510315]
    elif dataset == "pubmedqa":
        return [0.0,0.0,0.0,0.0,0.0,0.06087161973118782,0.9391283392906189,0.0,0.0,0.0]
    elif dataset == "triviaqa":
        return [0.0,0.0,0.0,0.6801438927650452,0.11240127682685852,0.0,0.2074548453092575,0.0,0.0,0.0]
    elif dataset == "truthfulqa_gen":
        return [0.0,0.0,0.0,0.0,0.0,0.20507599413394928,0.0,0.0,0.0,0.7949240207672119]
    else:   # fallback for wikitext, mmlu, ai2_arc
        return [1,0,0,0,0,0,0,0,0,0]

def sum_to_one(x: List[float]) -> List[float]:
    t = np.asarray(x, dtype=np.float32)
    s = float(t.sum())
    if s <= 1e-12:
        return (np.ones_like(t) / len(t)).tolist()
    return (t / s).tolist()

# Function to scale original raw params (18D) to [0,1] range, and the inverse
def scale_params(vec: List[float],
                 len_domains: int,
                 lora_max_layers: int = 32,
                 max_rank: int = 128,
                 direction: str = "forward") -> List[float]:
    
    v = list(vec)
    D = len_domains

    if direction == "forward":
        # Normalise data mix ratio
        mix = sum_to_one(v[:D])

        # Scale layers by max_layers
        layers_scaled = 0.0 if lora_max_layers <= 0 else np.clip(v[D] / float(lora_max_layers), 0.0, 1.0)

        # Ensure flags are 0/1
        flags = [float(1.0 if f >= 0.5 else 0.0) for f in v[D+1:D+6]]

        # Scale rank by max_rank
        rank_scaled = float(v[D+6]) / float(max_rank)

        # Scale dropout
        dropout_scaled = np.clip(v[D+7] / 0.1, 0.0, 1.0)

        return mix + [float(layers_scaled)] + flags + [float(rank_scaled)] + [float(dropout_scaled)]

    elif direction == "inverse":
        # clamp to [0,1]
        x_0_to_1 = np.clip(np.asarray(v, dtype=np.float32), 0.0, 1.0)
        
        # Normalise data mix ratio
        mix = sum_to_one(x_0_to_1[:D])

        # Scale layers by max_layers
        layers = int(round(float(x_0_to_1[D]) * float(lora_max_layers)))
        layers = int(np.clip(layers, 0, lora_max_layers))

        # Ensure flags are 0/1
        flags = [int(1 if z >= 0.5 else 0) for z in x_0_to_1[D+1:D+6]]
        if sum(flags) == 0:
            # force the strongest (largest u) to 1
            j = int(np.argmax(x_0_to_1[D+1:D+6]))
            flags[j] = 1

        # Scale rank by max_rank
        rank_raw = max(1, int(round(float(x_0_to_1[D+6]) * float(max_rank))))

        # Scale dropout
        drop = float(np.clip(x_0_to_1[D+7], 0.0, 1.0)) * 0.1

        return mix + [layers] + flags + [rank_raw] + [drop]

    else:
        raise ValueError("direction must be 'forward' or 'inverse'")

# SimpleVAE class to decode/encode latent space
class SimpleVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 10, hidden: int = 128, mixture_dim: int = 10):
        super().__init__()
        self.mixture_dim = mixture_dim
        rest_dim = input_dim - mixture_dim  # 8: layers01, 5 flags, rank01, drop01

        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)

        self.dec_core = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.dec_mix  = nn.Linear(hidden, mixture_dim)   # logits -> softmax
        self.dec_rest = nn.Linear(hidden, rest_dim)      # -> sigmoid

    def encode(self, x):
        h = self.enc(x);  return self.mu(h), self.logvar(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        h = self.dec_core(z)
        mix = F.softmax(self.dec_mix(h), dim=-1)     # sums to 1
        rest = torch.sigmoid(self.dec_rest(h))       # in [0,1]
        return torch.cat([mix, rest], dim=-1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        return self.decode(z), mu, logvar

# Train the VAE, given training_data (scaled to [0,1])
def train_vae(training_data: torch.Tensor,
              latent_dim: int = 10,
              hidden: int = 128,
              epochs: int = 50,
              lr: float = 1e-3,
              device: Optional[str] = None) -> SimpleVAE:

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleVAE(training_data.shape[1], latent_dim, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    training_data = training_data.to(device)
    model.train()
    for ep in range(epochs):
        xhat, mu, logvar = model(training_data)
        recon = F.mse_loss(xhat, training_data)
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + 0.2 * kld   # light KL
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    return model

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import math

# Decode BO output from unbounded VAE space, to [0,1] space, and finally to actual parameters values
@torch.no_grad()
def decode_to_config(
    vae: SimpleVAE,
    curr_vae_space: torch.Tensor,
    # latent_scaler: MinMaxScaler,       
    len_domains: int,
    lora_max_layers: int,
    rank_max: int = 128
) -> Tuple[List[float], List[float]]:
    
    # Step 1: Get GP_output
    gp_output = torch.tensor(curr_vae_space, dtype=torch.float32, device=next(vae.parameters()).device)

    # Step 2: 
    X_between_0_to_1 = vae.decode(gp_output.unsqueeze(0)).squeeze(0).detach().cpu().numpy().tolist()

    # DEBUG: Sanity check for VAE decoder output
    print("Decoded VAE latent before inverse_scaling: ", X_between_0_to_1)

    # ensure mixture sums to 1 before inverse scaling
    X_between_0_to_1[:len_domains] = sum_to_one(X_between_0_to_1[:len_domains])

    # map back to raw domain (layers int, flags 0/1 with at least one 1, rank snapped, dropout ∈ [0,0.1])
    x_raw = scale_params(X_between_0_to_1, len_domains, lora_max_layers, rank_max, direction="inverse")
    return x_raw, X_between_0_to_1


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
    sampling_method: str = "random",
    train_epochs: int = 1,
    training_batch: int = 8,
    evaluation_batch: int = 4,
    printout: bool = True,
    max_steps: int = -1,
    eval_steps: int = 100,
    limit: int = 100,
    latent_dim: int = 10,
    vae_hidden: int = 64,
    vae_epochs: int = 20,
    vae_lr: float = 1e-3,
    # NEW:
    vae_pretrain_samples: int = 2000,         # how many random configs to pretrain on
    vae_model_dir: str = "./vae_models",     # where to save/load VAE checkpoints
    model_id: str = "LLM/llama_8b_instruct",
):
    train_datasets = []
    val_datasets = []
    for data_domain in data_domains:
        train_dataset, val_dataset = load_data(data_domain=data_domain)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    lora_max_num_layers = 32
    
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    len_domains = len(data_domains)
    input_X = (np.ones(len_domains) / len_domains).tolist()     # uniform over your actual number of domains
    input_X.append(int(lora_max_num_layers * 0.5))           # layers
    input_X += [1, 1, 1, 1, 1]                               # flags
    input_X.append(72)                                       # rank
    input_X.append(0.05)                                     # dropout

    input_X_between_0_1 = scale_params(
        input_X, len(data_domains), lora_max_num_layers, lora_rank_max, "forward"
    )
    

    all_influences = [] # not used currently
    for train_domain in data_domains:
        all_influences.append(None)
        #all_influences.append(torch.load("influence/"+str(train_domain)+"_training.pt"))

    # Pretrain the VAE on randomly generated datapoints, within the bounds of [0,1] for each dimension.
    len_domains_ = len(data_domains)
    vae_samples = []
    for _ in range(vae_pretrain_samples):
        mix = np.random.dirichlet(np.ones(len_domains_)).tolist()
        layers = int(np.random.randint(0, lora_max_num_layers + 1))
        flags = (np.random.rand(5) < 0.5).astype(int).tolist()
        if sum(flags) == 0:
            flags[np.random.randint(0, 5)] = 1
        rank = int(np.random.randint(1, 129))
        drop = float(np.random.rand()*0.1)
        xr = mix + [layers] + flags + [rank] + [drop]

        # Scale samples to [0,1]. VAE encodes [0,1] -> latent space
        vae_samples.append(scale_params(xr, len_domains_, lora_max_num_layers, lora_rank_max, "forward"))
    
    # Add the initial sample (bad mixing ratio and preset lora config)
    vae_samples.append(scale_params(input_X, len_domains_, lora_max_num_layers, lora_rank_max, "forward"))
    vae_training_data = torch.tensor(np.asarray(vae_samples, dtype=np.float32))

    device = "cuda" if (isinstance(evaluation_cuda, str) and evaluation_cuda.startswith("cuda")) and torch.cuda.is_available() else "cpu"
    vae = train_vae(vae_training_data, latent_dim=latent_dim, hidden=vae_hidden, epochs=vae_epochs, lr=vae_lr, device=device)

    with torch.no_grad():
        mu_all, _ = vae.encode(vae_training_data.to(next(vae.parameters()).device))  # (N, latent_dim)
    
    # 22 aug afternoon: try latent mapper to map latent values -> [0,1]
    mapper = VAEToUnitMapper().fit(mu_all, scale_multiplier=2.0, lower_bound=0.5) 

    # Train scaler to scale from latent space to [0,1], for GP_input
    # latent_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    # latent_scaler.fit(mu_all.detach().cpu().numpy())    

    # # 22 aug try scaler
    # lo = -3*np.ones(latent_dim)
    # hi =  3*np.ones(latent_dim)
    # latent_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    # latent_scaler.fit(np.vstack([lo, hi]))

    # ### Testing gpt scaler
    # mu_np = mu_all.detach().cpu().numpy()              # shape: (N, latent_dim)
    # m = mu_np.mean(axis=0)                             # per-dim mean
    # std = mu_np.std(axis=0)                            # per-dim std

    # # Choose a safe span around the mean; fallback ensures non-zero range
    # span = np.maximum(2.0 * std, 0.5)                  # >= 0.5 per dim
    # lo = m - span
    # hi = m + span

    # # Guard against any inf/nan (very rare, but cheap insurance)
    # lo = np.where(np.isfinite(lo), lo, m - 0.5)
    # hi = np.where(np.isfinite(hi), hi, m + 0.5)

    # latent_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    # # Fit the scaler on synthetic 2-row “dataset” that encodes the bounds.
    # latent_scaler.fit(np.vstack([lo, hi]))
    
    # ### Testing gpt scaler (END)

    
    # BO bounds
    lower_bound = [0.0] * latent_dim
    upper_bound = [1.0] * latent_dim
    bounds = torch.stack([torch.tensor(lower_bound, dtype=torch.double),
                          torch.tensor(upper_bound, dtype=torch.double)])

    # for each BO iteration
    GP_input = [] # X
    observed_output = [] # y

    # Encode initial input_X to VAE latent space, then scale to [0,1] using the latent_scaler


    for i in tqdm(range(BO_run)):
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        tokenizer, model = get_tokenizer_and_model(model_id = model_id)

        print("iteration: ", i)
        print("input_X: ", input_X)
        if printout:
            print("mixing data with method: ", sampling_method)
        
        # take the model related inputs and arrange them in a nice lora config file
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
            base_path = lora_path.rsplit('/', 1)[0] + '/'
            shutil.rmtree(base_path, ignore_errors=True)
            for task in evaluation_task:
                task_weight, metric = evaluation_task[task]
                perf = results["results"][task][metric]
                if task == "wikitext":
                    perf = - perf # we want to maximize the score, so for perplexity we maximize instead
                observed_performance += (perf * task_weight)
            lora_model.to("cpu")
        print("current iteration weighted performance: ", observed_performance)

        # Encode current candidate to [0,1]
        #input_X_between_0_1 = scale_params(input_X, len(data_domains), lora_max_num_layers, lora_rank_max, "forward")
        
        # Encoded current [0,1] candidate to VAE latent space (gp input)
        # with torch.no_grad():
        #     mu_x, _ = vae.encode(torch.tensor([input_X_between_0_1], dtype=torch.float32, device=next(vae.parameters()).device))
        #     print("Encoded VAE latent mu_x: ", mu_x)
        #     z_mean_np = mu_x.detach().cpu().numpy()  # shape (1, latent_dim)
        #     u_mean_np = latent_scaler.transform(z_mean_np)  # shape (1, latent_dim) in [0,1]
        #     u_mean = torch.from_numpy(u_mean_np[0]).double()
        
        # 22 aug afternoon: try latent mapper
        with torch.no_grad():
            mu_x, _ = vae.encode(torch.tensor([input_X_between_0_1], dtype=torch.float32,
                                            device=next(vae.parameters()).device))  # (1, d)
            u_mean = mapper.to_unit(mu_x.squeeze(0))  # (d,)
        current_gp_input = u_mean.double().tolist()
        # current_gp_input = u_mean.tolist()

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
        # A = [1.0] * latent_dim
        # x = list(range(latent_dim)) # A, x is passed as equality constraints for data mixture. since the ratio needs to sum to 1.
        
        # acq optimization tells us next candidate
        candidate, acq_value = optimize_acqf(
            UCB, bounds=bounds, q=1, num_restarts=20, raw_samples=1024
        )
        print("proposed candidate before processing:", candidate[0])
        
        # Convert GP_output to original raw params (18D)
        def process_values(values, data_domains_len):
            curr_gp_output = values.detach().cpu().float()

            # Step 1: Inverse mapping of GP_output ([0,1]) to VAE latent space
            curr_vae_space = mapper.from_unit(curr_gp_output.float())     # unbounded latent

            # Step 2: Decode from VAE latent space to 18D ([0,1]), then to 18D (original range)
            x_raw, y01 = decode_to_config(
                vae=vae,
                curr_vae_space=curr_vae_space,
                # latent_scaler=latent_scaler,
                len_domains=len(data_domains),
                lora_max_layers=lora_max_num_layers,
                rank_max=lora_rank_max,
            )
            print("proposed candidate after processing:", x_raw)
            return x_raw, y01
        
        # Input_X: Decoded candidate value in raw parameter format
        # Input_X_between_0_1: Decoded candidate value in [0,1] format
        input_X, input_X_between_0_1 = process_values(candidate[0], len(data_domains))
        
    return GP_input, observed_output, gp

# 26 Aug: Try to use Deep Kernel Learning (DKL) instead of VAE.

import torch
import torch.nn as nn

from gpytorch.kernels import Kernel, MaternKernel, RBFKernel, ScaleKernel
from gpytorch.utils.grid import ScaleToBounds
    

class FeatureModule(nn.Sequential):
    def __init__(self, dim_seq):
        super().__init__()
        assert len(dim_seq) >= 2
        for i in range(len(dim_seq) - 1):
            self.add_module(f'linear{i}', nn.Linear(dim_seq[i], dim_seq[i+1]))
            if i + 2 < len(dim_seq):
                self.add_module(f'relu{i}', nn.ReLU())

class DeepKernel(Kernel):
    def __init__(self, dim_seq, base_kernel: Kernel = None, freeze_nn: bool = False,
                 use_scale_to_bounds: bool = True):
        super().__init__()
        self.feature_module = FeatureModule(dim_seq=dim_seq)
        if freeze_nn:
            self.feature_module.requires_grad_(False)
        self.kernel = ScaleKernel(MaternKernel(nu=2.5)) if base_kernel is None else base_kernel
        self.scale_to_bounds = ScaleToBounds(-1., 1.) if use_scale_to_bounds else None
        self.feature_module = self.feature_module.to(dtype=torch.double)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure dtype/device match the feature module
        fm_dtype = next(self.feature_module.parameters()).dtype
        fm_device = next(self.feature_module.parameters()).device
        x_f = x.to(dtype=fm_dtype, device=fm_device)
        z = self.feature_module(x_f)
        if self.scale_to_bounds is not None:
            z = self.scale_to_bounds(z)
        return z

    def forward(self, x1, x2, diag=False, **params):
        x1_t = self._transform(x1)
        x2_t = self._transform(x2)
        return self.kernel.forward(x1=x1_t, x2=x2_t, diag=diag, **params)

@torch.no_grad()
def decode_to_config_dkl(
    curr_vae_space: torch.Tensor,         # shape: (input_dim,), values in [0,1]
    len_domains: int,
    lora_max_layers: int,
    rank_max: int = 128
) -> Tuple[List[float], List[float]]:
    # ensure 0..1 and renormalize mixture part
    X_between_0_to_1 = torch.clamp(curr_vae_space.detach().cpu().float(), 0.0, 1.0).tolist()
    X_between_0_to_1[:len_domains] = sum_to_one(X_between_0_to_1[:len_domains])
    x_raw = scale_params(X_between_0_to_1, len_domains, lora_max_layers, rank_max, direction="inverse")
    return x_raw, X_between_0_to_1

# def joint_opt_BO_LLM_with_dkl(
#     time_callback,
#     lora_rank_max: int,
#     data_domains: List[str],
#     random_dir: str,
#     BO_run: int,
#     total_data: int,
#     evaluation_cuda: str,
#     evaluation_task: dict,
#     ucb_beta: float,
#     sampling_method: str = "random",
#     train_epochs: int = 1,
#     training_batch: int = 8,
#     evaluation_batch: int = 4,
#     printout: bool = True,
#     max_steps: int = -1,
#     eval_steps: int = 100,
#     limit: int = 100,
#     # DKL options:
#     dkl_feature_dim: int = 8,
#     dkl_hidden: int = 64,
#     dkl_freeze_nn: bool = False,
#     model_id: str = "LLM/llama_8b_instruct",
# ):
#     train_datasets = []
#     val_datasets = []
#     for data_domain in data_domains:
#         train_dataset, val_dataset = load_data(data_domain=data_domain)
#         train_datasets.append(train_dataset)
#         val_datasets.append(val_dataset)

#     lora_max_num_layers = 32
    
#     torch.manual_seed(42)
#     np.random.seed(42)
#     random.seed(42)

#     # Build initial config
#     len_domains = len(data_domains)
#     input_X = (np.ones(len_domains) / len_domains).tolist()     # uniform over your actual number of domains

#     input_X.append(int(lora_max_num_layers * 0.5))           # layers
#     input_X += [1, 1, 1, 1, 1]                               # flags
#     input_X.append(72)                                       # rank
#     input_X.append(0.05)                                     # dropout

#     input_X_between_0_1 = scale_params(
#         input_X, len(data_domains), lora_max_num_layers, lora_rank_max, "forward"
#     )
    
#     all_influences = [] # not used currently
#     for train_domain in data_domains:
#         all_influences.append(None)
#         #all_influences.append(torch.load("influence/"+str(train_domain)+"_training.pt"))
#    # === BoTorch/GP containers ===
#     GP_input = []          # list of [d] in [0,1]
#     observed_output = []   # list of scalars

#     # === Bounds are over the full parameter dim (mixture D + 8) ===
#     input_dim = len(data_domains) + 8
#     assert len(input_X_between_0_1) == input_dim
#     bounds = torch.stack([
#         torch.zeros(input_dim, dtype=torch.double),
#         torch.ones(input_dim, dtype=torch.double),
#     ])

#     tokenizer, base_model_for_train = get_tokenizer_and_model(model_id=model_id)
#     for i in range(BO_run):
#         model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
#         tokenizer, model = get_tokenizer_and_model(model_id=model_id)

#         if printout:
#             print("iteration: ", i)
#             print("input_X: ", input_X)
#             print("mixing data with method: ", sampling_method)

#         lora_config = arrange_lora_config(
#             input_X[-2], input_X[-1], input_X[len(data_domains)],
#             input_X[len(data_domains)+1:len(data_domains)+6]
#         )
#         if lora_config is None:
#             observed_performance = 0.1
#         else:
#             print("Number of epochs: ", train_epochs)
#             path_to_final_model = extract_data_mixture_and_train(
#                 model=model, random_dir=random_dir, tokenizer=tokenizer, 
#                 train_datasets=train_datasets,  
#                 val_datasets=val_datasets, 
#                 data_domains=data_domains, 
#                 mixing_ratio=input_X[:len(data_domains)], 
#                 additional_info=[None for _ in data_domains],
#                 total_number_datapoints=total_data, 
#                 run_name="BO_run_" + str(i),
#                 method=sampling_method,
#                 train_epochs=train_epochs, 
#                 batch_size=training_batch,
#                 max_step=max_steps,
#                 lora_config=lora_config,
#                 eval_steps=eval_steps, callback=[time_callback]
#             )

#             # free gpu memory
#             with torch.no_grad():
#                 torch.cuda.empty_cache()
#             print("evaluating...")
#             # Eval (unchanged)
#             lora_path = path_to_final_model
#             config = PeftConfig.from_pretrained(lora_path)
#             base = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype='auto')
#             lora_model = PeftModel.from_pretrained(base, lora_path).to(evaluation_cuda)
#             tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True,)

#             observed_performance = 0
#             tasks = list(evaluation_task.keys())
#             results = evaluate_tasks(tasks, lora_model, tokenizer, evaluation_batch, few_shot=1, limit=limit)
#             print("deleting lora model after evaluation.")
#             base_path = lora_path.rsplit('/', 1)[0] + '/'
#             shutil.rmtree(base_path, ignore_errors=True)
#             for task in evaluation_task:
#                 task_weight, metric = evaluation_task[task]
#                 perf = results["results"][task][metric]
#                 if task == "wikitext":
#                     perf = -perf
#                 observed_performance += (perf * task_weight)
#             lora_model.to("cpu")

#         print("current iteration weighted performance: ", observed_performance)

#         # === Current GP input is simply the [0,1] parameterization ===
#         current_gp_input = torch.tensor(input_X_between_0_1, dtype=torch.double)
#         GP_input.append(current_gp_input.tolist())
#         observed_output.append(observed_performance)

#         # === Fit DKL GP ===
#         train_X = torch.tensor(GP_input, dtype=torch.double)                  # [N, d]
#         train_Y = torch.tensor(observed_output, dtype=torch.double).view(-1,1) # [N, 1]

#         # Deep kernel: small MLP feature extractor
#         dim_seq = [input_dim, dkl_hidden, dkl_hidden, dkl_feature_dim]
#         deep_kernel = DeepKernel(dim_seq=dim_seq, freeze_nn=dkl_freeze_nn)

#         gp = SingleTaskGP(train_X, train_Y, covar_module=deep_kernel, outcome_transform=Standardize(m=1))
#         mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
#         fit_gpytorch_mll(mll)

#         # # (sanity print)
#         # print("GP past observed values (should be between [0,1]): ", GP_input)
        
#         # use Bayesian Optimization's acq function to propose next candidate mixing parameter and score parameters for agents
#         # UCB = UpperConfidenceBound(gp, beta=ucb_beta) # the acq function
#         # A = [1.0] * len(data_domains)
#         # x = list(range(len(data_domains))) # A, x is passed as equality constraints for data mixture. since the ratio needs to sum to 1.
#         # candidate, acq_value = optimize_acqf(
#         #     UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=10,
#         #     equality_constraints = [(torch.tensor(x), torch.tensor(A), 1)] # edit this TODO.
#         # )
#         UCB = UpperConfidenceBound(gp, beta=ucb_beta)

#         len_domains = len(data_domains)
#         dtype = bounds.dtype
#         device = bounds.device

#         idx = torch.arange(len_domains, dtype=torch.long, device=device)           # indices (Long)
#         coef = torch.ones(len_domains, dtype=dtype, device=device)                 # coefficients (same dtype as bounds)
#         rhs  = torch.tensor(1.0, dtype=dtype, device=device)             # rhs (same dtype as bounds)

#         candidate, acq_value = optimize_acqf(
#             UCB,
#             bounds=bounds,
#             q=1,
#             num_restarts=5,
#             raw_samples=10,
#             equality_constraints=[(idx, coef, rhs)],
#         )
#         # acq optimization tells us next candidate
#         cand = candidate[0].detach().cpu().float()
#         print("proposed candidate before processing:", candidate[0])
        
#         # Convert GP_output to original raw params (18D)
#         def process_values(values, data_domains_len):
#             result = []
            
#             # Step 1: Squash first `data_domains_len` elements if less than 0.05
#             for v in values[:data_domains_len]:
#                 result.append(0 if v.item() < 0.05 else v)
            
#             # Step 2: lora layers
#             if len(values) > data_domains_len:
#                 result.append(round(lora_max_num_layers*values[data_domains_len].item()))
            
#             # Step 3: Round the next 5 elements: integer options
#             start = data_domains_len + 1
#             for v in values[start:start+5]:
#                 result.append(round(v.item()))
            
#             # Step 4: lora rank
#             if len(values) > start + 5:
#                 result.append(round(lora_rank_max * values[start + 5].item()))
            
#             # Step 5: drop out; unchanged
#             if len(values) > start + 6:
#                 result.append(values[start + 6].item())
#             print("proposed candidate after processing:", result)
#             return result
        
#         input_X, input_X_between_0_1 = decode_to_config_dkl(curr_vae_space=candidate[0], len_domains=len_domains, lora_max_layers=lora_max_num_layers, rank_max=lora_rank_max)
#         # input_X_between_0_1 = list(candidate[0])
#         # input_X = process_values(candidate[0], len(data_domains))

#         min_layers = 1
#         min_rank   = 1
#         min_drop   = 0.01
#         max_drop   = 0.20

#         layers = int(np.clip(input_X[len_domains], min_layers, lora_max_num_layers))
#         flags5 = [1 if v >= 0.5 else 0 for v in input_X[len_domains+1:len_domains+6]]
#         rank   = int(np.clip(input_X[len_domains+6], min_rank, lora_rank_max))
#         drop   = float(np.clip(input_X[len_domains+7], min_drop, max_drop))

#         input_X[len_domains]                     = layers
#         input_X[len_domains+1:len_domains+6]    = flags5
#         input_X[len_domains+6]                   = rank
#         input_X[len_domains+7]                   = drop

#     return GP_input, observed_output, gp    

# Fix deadlock
def joint_opt_BO_LLM_with_dkl(
    time_callback,
    lora_rank_max: int,
    data_domains: List[str],
    random_dir: str,
    BO_run: int,
    total_data: int,
    evaluation_cuda: str,
    evaluation_task: dict,
    ucb_beta: float,
    sampling_method: str = "random",
    train_epochs: int = 1,
    training_batch: int = 8,
    evaluation_batch: int = 4,
    printout: bool = True,
    max_steps: int = -1,
    eval_steps: int = 100,
    limit: int = 100,
    seed: int=42,
    # DKL options:
    dkl_feature_dim: int = 8,
    dkl_hidden: int = 64,
    dkl_freeze_nn: bool = False,
    model_id: str = "LLM/llama_8b_instruct",
):
    import os, gc  # new
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # new
    os.environ.setdefault("OMP_NUM_THREADS", "1")  # new
    os.environ.setdefault("MKL_NUM_THREADS", "1")  # new
    torch.set_num_threads(1)  # new

    train_datasets = []
    val_datasets = []
    for data_domain in data_domains:
        train_dataset, val_dataset = load_data(data_domain=data_domain)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    lora_max_num_layers = 32
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    len_domains = len(data_domains)
    # input_X = (np.ones(len_domains) / len_domains).tolist()
    input_X = get_mixing_ratio(evaluation_task)
    input_X.append(int(lora_max_num_layers * 0.5))
    input_X += [1, 1, 1, 1, 1]
    input_X.append(72)
    input_X.append(0.05)

    input_X_between_0_1 = scale_params(
        input_X, len(data_domains), lora_max_num_layers, lora_rank_max, "forward"
    )
    
    all_influences = []
    for train_domain in data_domains:
        all_influences.append(None)

    GP_input = []
    observed_output = []

    input_dim = len(data_domains) + 8
    assert len(input_X_between_0_1) == input_dim
    bounds = torch.stack([
        torch.zeros(input_dim, dtype=torch.double),
        torch.ones(input_dim, dtype=torch.double),
    ])
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer, base_model_cpu = get_tokenizer_and_model(model_id=model_id)  # new
    base_model_cpu = base_model_cpu.to("cpu")  # new

    for i in range(BO_run):
        if printout:
            print("iteration: ", i)
            print("input_X: ", input_X)
            print("mixing data with method: ", sampling_method)

        lora_config = arrange_lora_config(
            input_X[-2], input_X[-1], input_X[len(data_domains)],
            input_X[len(data_domains)+1:len(data_domains)+6]
        )
        if lora_config is None:
            observed_performance = 0.1
        else:
            print("Number of epochs: ", train_epochs)
            path_to_final_model = extract_data_mixture_and_train(
                model=base_model_cpu,  # new
                random_dir=random_dir, tokenizer=tokenizer, 
                train_datasets=train_datasets,  
                val_datasets=val_datasets, 
                data_domains=data_domains, 
                mixing_ratio=input_X[:len(data_domains)], 
                additional_info=[None for _ in data_domains],
                total_number_datapoints=total_data, 
                run_name=f"BO_run_{i}_{os.getpid()}",  # new
                method=sampling_method,
                train_epochs=train_epochs, 
                batch_size=training_batch,
                max_step=max_steps,
                lora_config=lora_config,
                eval_steps=eval_steps, callback=[time_callback]
            )

            with torch.no_grad():
                torch.cuda.empty_cache()
            print("evaluating...")
            lora_path = path_to_final_model
            lora_model = PeftModel.from_pretrained(base_model_cpu, lora_path).to(evaluation_cuda)  # new

            observed_performance = 0
            tasks = list(evaluation_task.keys())
            results = evaluate_tasks(tasks, lora_model, tokenizer, evaluation_batch, few_shot=1, limit=limit)
            print("deleting lora model after evaluation.")
            lora_model.to("cpu")  # new
            del lora_model  # new
            gc.collect()  # new
            torch.cuda.empty_cache()  # new
            shutil.rmtree(lora_path, ignore_errors=True)  # new
            for task in evaluation_task:
                task_weight, metric = evaluation_task[task]
                perf = results["results"][task][metric]
                if task == "wikitext":
                    perf = -perf
                observed_performance += (perf * task_weight)

        print("current iteration weighted performance: ", observed_performance)

        current_gp_input = torch.tensor(input_X_between_0_1, dtype=torch.double)
        GP_input.append(current_gp_input.tolist())
        observed_output.append(observed_performance)

        train_X = torch.tensor(GP_input, dtype=torch.double)
        train_Y = torch.tensor(observed_output, dtype=torch.double).view(-1,1)

        dim_seq = [input_dim, dkl_hidden, dkl_hidden, dkl_feature_dim]
        deep_kernel = DeepKernel(dim_seq=dim_seq, freeze_nn=dkl_freeze_nn)

        gp = SingleTaskGP(train_X, train_Y, covar_module=deep_kernel, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        
        UCB = UpperConfidenceBound(gp, beta=ucb_beta)

        len_domains = len(data_domains)
        dtype = bounds.dtype
        device = bounds.device

        idx = torch.arange(len_domains, dtype=torch.long, device=device)
        coef = torch.ones(len_domains, dtype=dtype, device=device)
        rhs  = torch.tensor(1.0, dtype=dtype, device=device)

        candidate, acq_value = optimize_acqf(
            UCB,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=10,
            equality_constraints=[(idx, coef, rhs)],
        )
        cand = candidate[0].detach().cpu().float()
        print("proposed candidate before processing:", candidate[0])
        
        def process_values(values, data_domains_len):
            result = []
            for v in values[:data_domains_len]:
                result.append(0 if v.item() < 0.05 else v)
            if len(values) > data_domains_len:
                result.append(round(lora_max_num_layers*values[data_domains_len].item()))
            start = data_domains_len + 1
            for v in values[start:start+5]:
                result.append(round(v.item()))
            if len(values) > start + 5:
                result.append(round(lora_rank_max * values[start + 5].item()))
            if len(values) > start + 6:
                result.append(values[start + 6].item())
            print("proposed candidate after processing:", result)
            return result
        
        input_X, input_X_between_0_1 = decode_to_config_dkl(curr_vae_space=candidate[0], len_domains=len_domains, lora_max_layers=lora_max_num_layers, rank_max=lora_rank_max)

        min_layers = 1
        min_rank   = 1
        min_drop   = 0.01
        max_drop   = 0.20

        layers = int(np.clip(input_X[len_domains], min_layers, lora_max_num_layers))
        flags5 = [1 if v >= 0.5 else 0 for v in input_X[len_domains+1:len_domains+6]]
        rank   = int(np.clip(input_X[len_domains+6], min_rank, lora_rank_max))
        drop   = float(np.clip(input_X[len_domains+7], min_drop, max_drop))

        input_X[len_domains]                     = layers
        input_X[len_domains+1:len_domains+6]    = flags5
        input_X[len_domains+6]                   = rank
        input_X[len_domains+7]                   = drop

    return GP_input, observed_output, gp