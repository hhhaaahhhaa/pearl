import logging
import sys
import os
import numpy as np
import torch
from tqdm import tqdm
from typing import Optional
import pickle

from actor import Actor
from env import Env
from llm import LLM
from datamodule import DataModule
from eval_agents import *

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')


def evaluate(dataset, agent: BaseAgent, log_file: Optional[str]=None):
    logs = []
    acc, total_length = 0, 0
    n = len(dataset)
    for sample in tqdm(dataset):
        res = agent.eval(sample)
        acc += res["is_correct"]
        total_length += res["response_length"]
        logs.append(res["log"])
    
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'wb') as f:
            pickle.dump(logs, f)
            # print(logs)
    return acc / n, total_length / n


def get_baselines(model_name: str, output_dir: str, debug=False):
    """ Baseline Evaluations """
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    datamodule = DataModule()

    # Debug with small portion of samples
    if debug:
        test_dataset = datamodule.test_dataset[:10]
    else:
        test_dataset = datamodule.test_dataset

    llm = LLM(model_name)
    tokenizer = llm.tokenizer
    model = llm.model
    env = Env(model, tokenizer, datamodule=datamodule)  # use baseclass since evaluation will not use the reward function
    prompt_pool = env.actions_str

    # No prompt baseline
    os.makedirs(f"{output_dir}/no_prompt", exist_ok=True)
    eval_agent = NoPromptAgent(llm)
    acc, avg_length = evaluate(test_dataset, eval_agent, log_file=f"{output_dir}/no_prompt/no-prompt-log.pkl")
    with open(f"{output_dir}/no_prompt/results.txt", "w") as f:
        f.write(f"Acc: {acc * 100:.2f}%, Avg length: {avg_length:.2f}")

    # Fix prompt baselines
    os.makedirs(f"{output_dir}/fix/logs", exist_ok=True)
    os.makedirs(f"{output_dir}/fix/results", exist_ok=True)
    for idx, prompt in enumerate(prompt_pool):
        eval_agent = FixPromptAgent(llm, prompt=prompt)
        acc, avg_length = evaluate(test_dataset, eval_agent, log_file=f"{output_dir}/fix/logs/prompt-{idx}.pkl")
        with open(f"{output_dir}/fix/results/prompt-{idx}.txt", "w") as f:
            f.write(f"Acc: {acc * 100:.2f}%, Avg length: {avg_length:.2f}")

    # Random baseline & Topline, directly computed from previous logs
    all_scores = {}
    for idx in range(len(prompt_pool)):
        with open(f"{output_dir}/fix/logs/prompt-{idx}.pkl", 'rb') as f:
            all_scores[idx] = pickle.load(f)

    random.seed(666)
    os.makedirs(f"{output_dir}/random", exist_ok=True)
    acc, total_length = 0, 0
    for i, sample in tqdm(enumerate(test_dataset)):
        k = random.randint(0, len(prompt_pool) - 1)
        prompt = prompt_pool[k]
        response = sample["prompt"][prompt]
        choices_prob = all_scores[k][i]["scores"]
        label = sample["options"].index(sample["answer"])
        acc += (np.argmax(choices_prob).item() == label)
        total_length += len(llm.tokenizer.tokenize(response))
    with open(f"{output_dir}/random/results.txt", "w") as f:
        f.write(f"Acc: {acc / len(test_dataset) * 100:.2f}%, Avg length: {total_length / len(test_dataset):.2f}")
    
    os.makedirs(f"{output_dir}/topline", exist_ok=True)
    acc, total_length, total_min_length, total_max_length = 0, 0, 0, 0
    for i, sample in tqdm(enumerate(test_dataset)):
        label = sample["options"].index(sample["answer"])
        temp_acc = 0
        correct_prompt_lengths, all_prompt_lengths = [], []
        for k in range(len(prompt_pool)):
            prompt = prompt_pool[k]
            response = sample["prompt"][prompt]
            choices_prob = all_scores[k][i]["scores"]
            actual_response_length = len(llm.tokenizer.tokenize(response))
            all_prompt_lengths.append(actual_response_length)
            if np.argmax(choices_prob).item() == label:
                temp_acc = 1
                prompt = prompt_pool[k]
                response = sample["prompt"][prompt]
                correct_prompt_lengths.append(actual_response_length)
        acc += temp_acc
        if len(correct_prompt_lengths) == 0:  # none of prompt solve the question, therefore the best prompt can be any possible prompt
            correct_prompt_lengths = all_prompt_lengths
        total_length += sum(correct_prompt_lengths) / len(correct_prompt_lengths)
        total_min_length += min(correct_prompt_lengths)
        total_max_length += max(correct_prompt_lengths)
    with open(f"{output_dir}/topline/results.txt", "w") as f:
        f.write(f"Acc: {acc / len(test_dataset) * 100:.2f}%, " + 
                f"Avg length: {total_length / len(test_dataset):.2f}, " + 
                f"Avg min length: {total_min_length / len(test_dataset):.2f}, " + 
                f"Avg max length: {total_max_length / len(test_dataset):.2f}")


def eval_pearl(model_name: str, checkpoint_dir: str, output_dir: str, debug=False):
    """ Evaluation for pearl after training """
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    datamodule = DataModule()

    # Debug with small portion of samples
    if debug:
        test_dataset = datamodule.test_dataset[:10]
    else:
        test_dataset = datamodule.test_dataset

    llm = LLM(model_name)
    tokenizer = llm.tokenizer
    model = llm.model
    env = Env(model, tokenizer, datamodule=datamodule, model_name=model_name)  # use baseclass since evaluation will not use the reward function
    prompt_pool = env.actions_str

    # Load checkpoint here
    actor = Actor(env, model, tokenizer)
    agent = actor.agent_ppo(update_interval=10, minibatch_size=1, epochs=20)
    print(f"Load from {checkpoint_dir}...")
    agent.load(checkpoint_dir)

    # Pearl Agent
    eval_agent = PearlAgent(llm, actor, prompt_pool)
    acc, avg_length = evaluate(test_dataset, eval_agent, log_file=f"{output_dir}/logs/log.pkl")
    with open(f"{output_dir}/results.txt", "w") as f:
        f.write(f"Acc: {acc * 100:.2f}%, Avg length: {avg_length:.2f}")


if __name__ == "__main__":
    # get_baselines("gpt2", "_baselines1", debug=True)
    # get_baselines("gpt2", "_baselines", debug=False)
    eval_pearl("gpt2", "somewhere/1000_finish", "_exp-debug", debug=True)
    # eval_pearl("meta-llama/Llama-2-7b-chat-hf", "_ray_checkpoints/prob_divided_by_token_length", "_exp/prob_divided_by_token_length")
