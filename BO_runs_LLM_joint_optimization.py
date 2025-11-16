import json
from BO import run_BO_for_LLM, joint_opt_BO_LLM, joint_opt_BO_LLM_only_model, joint_opt_random, joint_opt_BO_LLM_only_data, joint_opt_BO_LLM_fixed_feature_list, evaluate_single_configuration, joint_opt_BO_LLM_with_vae, joint_opt_BO_LLM_interleave

from argparse import ArgumentParser
from transformers import TrainerCallback
import time


run_name = "BO_runs_LLM_joint_optimization"
parser = ArgumentParser()
parser.add_argument("--contaminate", help="to contaminate training data?", type=int, default=0) # WIP. not used.
parser.add_argument("--iterations", help="iterations BO?", type=int, default=10)
parser.add_argument("--num_data", help="total_data?", type=int, default=10000)
parser.add_argument("--epochs", help="epochs", default=1)
parser.add_argument("--trials", help="trials", default=1)
parser.add_argument("--evaluation_cuda", help="evaluation_cuda", default=0)
parser.add_argument("--sample_method", help="sample_method", default="random") # random, IF_random, IF_remove_harmful. But default to random
parser.add_argument("--eval_tasks", help="eval_tasks") # see task_metrics for different tasks
parser.add_argument("--experiments_setting", help="either ood or in_dist")
parser.add_argument("--output_dir", help="output_dir")
parser.add_argument("--time_limit", help="time_limit")
parser.add_argument("--lora_rank", help="lora_rank")
parser.add_argument("--ucb_beta", help="lora_rank")
parser.add_argument("--limit", help="no. of samples for performance evaluation. Default is 100", default=100)
parser.add_argument("--run_BO_on", help="all or model or data or single_eval", default="all")
parser.add_argument("--seed", help="seed value for single eval", type=int)
parser.add_argument("--model")
parser.add_argument("--init_mixing_ratio", help="Only for manual input: initial mixing ratio, comma separated, len=number of data domains", default=None) # e.g. 0.1,0.2,0.3...
parser.add_argument("--init_lora_num_layers", help="Only for  manual input: initial lora num layers", default=None) # e.g. 1
parser.add_argument("--init_lora_modules", help="Only for  manual input: initial lora modules, comma separated, len=5", default=None) # e.g. 1,0,0,1,1
parser.add_argument("--init_lora_rank", help="Only for  manual input: initial lora rank", default=None) # e.g. 16
parser.add_argument("--init_lora_dropout", help="Only for  manual input: initial lora dropout", default=None) # e.g. 0.05
parser.add_argument("--iteration", help="Only for  manual input, extrapolation purposes", default=1) # e.g. 1
parser.add_argument("--idx", help="Only for  manual input, extrapolation purposes eg. trial_{idx}.json", default=1) # e.g. 1
parser.add_argument("--vae_dim", help="latent dim for VAE-BO", type=int, default=10)
parser.add_argument(
    "--vae_hidden",
    type=int,
    default=64,
    help="Dimensionality of the VAE’s hidden layers."
)
parser.add_argument(
    "--vae_epochs",
    type=int,
    default=20,
    help="Number of epochs to train the VAE."
)
parser.add_argument(
    "--vae_lr",
    type=float,
    default=1e-3,
    help="Learning rate for the VAE optimizer."
)
parser.add_argument("--dkl_feature_dim", help="dkl feature dim", type=int, default=32)
parser.add_argument("--dkl_hidden", help="dkl hidden layers", type=int, default=64)
parser.add_argument("--dkl_freeze_nn", help="dkl freeze nn", type=bool, default=False)

class TimerCallback(TrainerCallback):
    def __init__(self, max_duration_seconds):
        self.max_duration = int(max_duration_seconds)
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time
        if elapsed >= self.max_duration:
            print(f"⏰ Max training time of {self.max_duration} seconds reached. Stopping.")
            control.should_training_stop = True
        return control
    
args = vars(parser.parse_args())
print("command-line args: ", args)

# WIP
to_contaminate= bool(args["contaminate"])
if not to_contaminate:
    influence_path="influence/"
    print("getting influence from: ", influence_path)
else:
    influence_path="influence/contaminated/"
    print("getting influence from: ", influence_path)

setting=(args["experiments_setting"])
output_dir=(args["output_dir"])
time_limit = int(args["time_limit"])
epochs=int(args["epochs"])
trials=int(args["trials"])
cuda=int(args["evaluation_cuda"])
cuda="cuda:"+str(cuda)
BO_run = int(args["iterations"])
total_data = int(args["num_data"])
sample_methods = str(args["sample_method"]).split(",")
tasks = str(args["eval_tasks"]).split(",") # list of eval task
evaluation_weights = [1/len(tasks)] * len(tasks)
lora_rank =int(args["lora_rank"])
ucb_beta = float(args["ucb_beta"])
run_BO_on = str(args["run_BO_on"]) # either "model" or "data"
limit = int(args["limit"]) # either "model" or "data" or "all"
model_id = str(args["model"])
if args["init_mixing_ratio"] is not None:
    init_mixing_ratio = [float(x) for x in args["init_mixing_ratio"].split(",")]
else:
    init_mixing_ratio = None
if args["init_lora_num_layers"] is not None:
    init_lora_num_layers = int(args["init_lora_num_layers"])
else:
    init_lora_num_layers = None
if args["init_lora_modules"] is not None:
    init_lora_modules = [int(x) for x in args["init_lora_modules"].split(",")]
else:
    init_lora_modules = None
if args["init_lora_rank"] is not None:
    init_lora_rank = int(args["init_lora_rank"])
else:
    init_lora_rank = None
if args["init_lora_dropout"] is not None:
    init_lora_dropout = float(args["init_lora_dropout"])
else:
    init_lora_dropout = None

if limit < 0:
    limit = None

import random
import string
# Generate a random string of size 5 using uppercase and lowercase letters
random_string = ''.join(random.choices(string.ascii_letters, k=5))
print("random sentence created:", random_string)

task_metrics = {
  "commonsense_qa": "acc,none",
  "gsm8k": "exact_match,strict-match",
  "headqa_en": "acc,none",
  "rowan_hellaswag": "acc,none",
  "pubmedqa": "acc,none",
  "sciq": "acc_norm,none",
  "triviaqa": "exact_match,remove_whitespace",
  "truthfulqa_gen": "bleu_acc,none",
  "wikitext": "word_perplexity,none",
  "mmlu": "acc,none",
  "ai2_arc": "acc,none"
}

# set up training data (depending if we want ood)
data_domains_initial = list(task_metrics.keys())
print("current eval task: ", tasks)
if setting == "ood":
    data_domains =  [x for x in data_domains_initial if x not in tasks] # remove training domain that is in task
else:
    data_domains = [x for x in data_domains_initial]

# set up evaluation tasks (and weights, if we have more than one evaluation task)
evaluation_task = {}
for task, weight in zip(tasks, evaluation_weights):
    evaluation_task[task] = (float(weight), task_metrics[task])

print("evaluation tasks and weights: ", evaluation_task)

train_epochs = 1
training_batch = 8
evaluation_batch = 4
evaluation_steps=50000
final_info_stored = {"command line args": args,
                    "hash string": random_string,
                    "training domain": data_domains,
                    "evaluation domain": tasks,
                    "weight": evaluation_weights} # weight in str

for sample_method in sample_methods: # random sampling
    results = []
    print("num of trials: ", trials)
    for x in range(trials):
        #model_id="Qwen/Qwen2.5-7B-Instruct" # pass this into next function if necessary
        #model_id: str = "LLM/llama_8b_instruct"
        
        # rng = random.Random()
        # seed = rng.randint(0, 1000)

        seed = x

        if run_BO_on == "all": # run BO on both data and model
            print("running BO on both data and model")
            GP_input, observed_output, gp = joint_opt_BO_LLM(time_callback=TimerCallback(time_limit), lora_rank_max=lora_rank, data_domains = data_domains,
                                                        random_dir = random_string, 
                                                            BO_run = BO_run,
                                                            total_data = total_data, 
                                                            evaluation_cuda = cuda, 
                                                            evaluation_task = evaluation_task,
                                                            trial_number=x,
                                                            sampling_method = sample_method, 
                                                            train_epochs=train_epochs, 
                                                            training_batch=training_batch, 
                                                            evaluation_batch=evaluation_batch,
                                                            eval_steps=evaluation_steps,
                                                            ucb_beta=ucb_beta,
                                                            limit=limit,
                                                            printout=True,
                                                            seed=args["seed"],
                                                            output_dir=output_dir)
        # elif run_BO_on == "JoBS":
        #     print("running BO with DKL")
        #     GP_input, observed_output, gp = joint_opt_BO_LLM_with_dkl(time_callback=TimerCallback(time_limit),
        #                                                             lora_rank_max=lora_rank, data_domains=data_domains,
        #                                                             random_dir=random_string, BO_run=BO_run, total_data=total_data,
        #                                                             evaluation_cuda=cuda, evaluation_task=evaluation_task,
        #                                                             ucb_beta=ucb_beta, sampling_method=sample_method,
        #                                                             train_epochs=train_epochs, training_batch=training_batch,
        #                                                             evaluation_batch=evaluation_batch, printout=True,
        #                                                             max_steps=evaluation_steps, eval_steps=evaluation_steps,
        #                                                             limit=limit, seed=seed, output_dir=output_dir,
        #                                                             dkl_feature_dim=args["dkl_feature_dim"], dkl_hidden=args["dkl_hidden"],dkl_freeze_nn=args["dkl_freeze_nn"])
        elif run_BO_on == "all_fixed_features": # run BO on both data and model, but with some discrete optimization tricks; somehow this doesn't work well.
            print("running BO on both data and model with fixed feature list")
            GP_input, observed_output, gp = joint_opt_BO_LLM_fixed_feature_list(time_callback=TimerCallback(time_limit), lora_rank_max=lora_rank, data_domains = data_domains,
                                                        random_dir = random_string, 
                                                            BO_run = BO_run,
                                                            total_data = total_data, 
                                                            evaluation_cuda = cuda, 
                                                            evaluation_task = evaluation_task,
                                                            sampling_method = sample_method, 
                                                            train_epochs=train_epochs, 
                                                            training_batch=training_batch, 
                                                            evaluation_batch=evaluation_batch,
                                                            eval_steps=evaluation_steps,
                                                            ucb_beta=ucb_beta,
                                                            limit=limit,
                                                            printout=True)
        elif run_BO_on == "model": # run BO only on model
            print("running BO only on model")
            GP_input, observed_output, gp = joint_opt_BO_LLM_only_model(time_callback=TimerCallback(time_limit), lora_rank_max=lora_rank, data_domains = data_domains,
                                                        random_dir = random_string, 
                                                            BO_run = BO_run,
                                                            total_data = total_data, 
                                                            evaluation_cuda = cuda, 
                                                            evaluation_task = evaluation_task,
                                                            trial_number=x,
                                                            sampling_method = sample_method, 
                                                            train_epochs=train_epochs, 
                                                            training_batch=training_batch, 
                                                            evaluation_batch=evaluation_batch,
                                                            eval_steps=evaluation_steps,
                                                            ucb_beta=ucb_beta,
                                                            limit=limit,
                                                            printout=True,
                                                            seed=seed,
                                                            output_dir=output_dir)
        elif run_BO_on == "data":
            print("running BO only on data")
            GP_input, observed_output, gp = joint_opt_BO_LLM_only_data(time_callback=TimerCallback(time_limit), default_alpha=16, default_dropout=0.05, default_layer=[1,1,1,1,1],
                                                                       default_num_layers_to_apply=14, default_rank=72,
                                                                       data_domains = data_domains,
                                                                        random_dir = random_string, 
                                                                        BO_run = BO_run,
                                                                        total_data = total_data, 
                                                                        evaluation_cuda = cuda, 
                                                                        evaluation_task = evaluation_task,
                                                                        trial_number=x,
                                                                        sampling_method = sample_method, 
                                                                        train_epochs=train_epochs, 
                                                                        training_batch=training_batch, 
                                                                        evaluation_batch=evaluation_batch,
                                                                        eval_steps=evaluation_steps,
                                                                        ucb_beta=ucb_beta,
                                                                        limit=limit,
                                                                        printout=True,
                                                                        seed=args["seed"],
                                                                        output_dir=output_dir)
        elif run_BO_on == "random":
            print("using random configurations")
            joint_opt_random(lora_rank_max=lora_rank, data_domains = data_domains,
                                                        random_dir = random_string, 
                                                            BO_run = BO_run,
                                                            total_data = total_data, 
                                                            evaluation_cuda = cuda, 
                                                            evaluation_task = evaluation_task,
                                                            sampling_method = sample_method, 
                                                            train_epochs=epochs, 
                                                            training_batch=training_batch, 
                                                            evaluation_batch=evaluation_batch,
                                                            eval_steps=evaluation_steps,
                                                            ucb_beta=ucb_beta,
                                                            limit=limit,
                                                            printout=True,
                                                            seed=args["seed"],
                                                            output_dir=output_dir,
                                                            trial_number=x,
                                                            time_limit=time_limit,
                                                            model_id=model_id)
            exit()  # exit after random run
        elif run_BO_on == "vae":
            print("running BO with VAE")
            GP_input, observed_output, gp = joint_opt_BO_LLM_with_vae(time_callback=TimerCallback(time_limit),
                                                                    lora_rank_max=lora_rank, 
                                                                    data_domains=data_domains,
                                                                    random_dir=random_string, 
                                                                    BO_run=BO_run, 
                                                                    total_data=total_data,
                                                                    evaluation_cuda=cuda, 
                                                                    evaluation_task=evaluation_task,
                                                                    trial_number=x,
                                                                    ucb_beta=ucb_beta, 
                                                                    sampling_method=sample_method,
                                                                    train_epochs=train_epochs, 
                                                                    training_batch=training_batch,
                                                                    evaluation_batch=evaluation_batch, 
                                                                    printout=True,
                                                                    max_steps=evaluation_steps, 
                                                                    eval_steps=evaluation_steps,
                                                                    limit=limit,
                                                                    seed=seed,
                                                                    output_dir=output_dir,
                                                                    latent_dim=args["vae_dim"], 
                                                                    vae_hidden=args["vae_hidden"],
                                                                    vae_epochs= args["vae_epochs"], 
                                                                    vae_lr=args["vae_lr"])
        elif run_BO_on == "single_eval":    # this is for manual input, not running BO
            print("not running BO, just printing performance when using manual inputs")
            observed_output = evaluate_single_configuration(time_callback=TimerCallback(time_limit), lora_rank_max=lora_rank, data_domains = data_domains,
                                                            random_dir = random_string, 
                                                            total_data = total_data, 
                                                            evaluation_cuda = cuda, 
                                                            evaluation_task = evaluation_task,
                                                            seed=args["seed"],
                                                            # seed=seed,
                                                            sampling_method = sample_method, 
                                                            train_epochs=train_epochs, 
                                                            training_batch=training_batch, 
                                                            evaluation_batch=evaluation_batch,
                                                            eval_steps=evaluation_steps,
                                                            limit=limit,
                                                            init_mixing_ratio=init_mixing_ratio,
                                                            init_lora_num_layers=init_lora_num_layers,
                                                            init_lora_modules=init_lora_modules,
                                                            init_lora_rank=init_lora_rank,
                                                            init_lora_dropout=init_lora_dropout,
                                                            iteration=int(args["iteration"]),
                                                            trial_idx = args["idx"],
                                                            output_dir=output_dir,
                                                            )
            exit() # Exit after single eval, no need to continue
        elif run_BO_on == "interleave":
            print("running interleaved BO on both data and model")
            GP_input, observed_output, gp = joint_opt_BO_LLM_interleave(time_callback=TimerCallback(time_limit), lora_rank_max=lora_rank, data_domains = data_domains,
                                                        random_dir = random_string, 
                                                            BO_run = BO_run,
                                                            total_data = total_data, 
                                                            evaluation_cuda = cuda, 
                                                            evaluation_task = evaluation_task,
                                                            trial_number=x,
                                                            sampling_method = sample_method, 
                                                            train_epochs=train_epochs, 
                                                            training_batch=training_batch, 
                                                            evaluation_batch=evaluation_batch,
                                                            eval_steps=evaluation_steps,
                                                            ucb_beta=ucb_beta,
                                                            limit=limit,
                                                            printout=True,
                                                            seed=seed,
                                                            output_dir=output_dir)
        else:
            assert False

        current_max = float('-inf')  # Start with negative infinity
        max_until_now = []           # List to store max values at each step

        # Iterate through the list
        for num in observed_output:
            current_max = max(current_max, num)  # Update the current maximum
            max_until_now.append(current_max)    # Store the max up to this step

        # best performance seen by BO at every step
        print("Best at every step:", max_until_now)
        results.append(max_until_now)
    final_info_stored[sample_method] = results
print("final results: ", final_info_stored)

# store results
# try:
#     with open("LLM/BO/" + output_dir + "/" +run_name+ "_".join(tasks)+".json", 'w') as f:
#         json.dump(final_info_stored, f)
# except:
#     print("error with storing json, it's ok. moving to next...")





