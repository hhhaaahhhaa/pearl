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


def split(model_name: str, output_dir: str):
    # Setup
    datamodule = DataModule()
    for split_name in datamodule.test_split_names:
        os.makedirs(f"{output_dir}/splits/{split_name}", exist_ok=True)
    llm = LLM(model_name)  # hack! We do not need to load llama, but need to load tokenizer
    tokenizer = llm.tokenizer
    model = llm.model
    env = Env(model, tokenizer, datamodule=datamodule)  # use baseclass since evaluation will not use the reward function
    prompt_pool = env.actions_str

    # No prompt
    with open(f"{output_dir}/no_prompt/no-prompt-log.pkl", 'rb') as f:
        no_prompt_scores = pickle.load(f)
    info = {
        x: {
            "acc": 0.0,
            "total_length": 0.0,
            "cnt": 0,
            "log": [],
        }
    for x in datamodule.test_split_names}
    for i, sample in tqdm(enumerate(datamodule.test_dataset)):
        split_name = sample["split_name"]
        choices_prob = no_prompt_scores[i]["scores"]
        label = sample["options"].index(sample["answer"])
        info[split_name]["acc"] += (np.argmax(choices_prob).item() == label)
        info[split_name]["total_length"] += 0
        info[split_name]["cnt"] += 1
        info[split_name]["log"].append(no_prompt_scores[i])
    
    # Create log
    for split_name in datamodule.test_split_names:
        os.makedirs(f"{output_dir}/splits/{split_name}/no_prompt", exist_ok=True)
        with open(f"{output_dir}/splits/{split_name}/no_prompt/results.txt", "w") as f:
            f.write(f"Acc: {info[split_name]['acc'] / info[split_name]['cnt'] * 100:.2f}%, Avg length: {info[split_name]['total_length'] / info[split_name]['cnt']:.2f}")
        with open(f"{output_dir}/splits/{split_name}/no_prompt/no-prompt-log.pkl", 'wb') as f:
            assert len(info[split_name]["log"]) == 500
            pickle.dump(info[split_name]["log"], f)

    # Fix prompt
    all_scores = {}
    for idx in range(len(prompt_pool)):
        with open(f"{output_dir}/fix/logs/prompt-{idx}.pkl", 'rb') as f:
            all_scores[idx] = pickle.load(f)
    for k, prompt in enumerate(prompt_pool):
        info = {
            x: {
                "acc": 0.0,
                "total_length": 0.0,
                "cnt": 0,
                "log": [],
            }
        for x in datamodule.test_split_names}
        for i, sample in tqdm(enumerate(datamodule.test_dataset)):
            split_name = sample["split_name"]
            prompt = prompt_pool[k]
            response = sample["prompt"][prompt]
            choices_prob = all_scores[k][i]["scores"]
            label = sample["options"].index(sample["answer"])
            info[split_name]["acc"] += (np.argmax(choices_prob).item() == label)
            info[split_name]["total_length"] += len(llm.tokenizer.tokenize(response))
            info[split_name]["cnt"] += 1
            info[split_name]["log"].append(all_scores[k][i])
        
        # Create log
        for split_name in datamodule.test_split_names:
            os.makedirs(f"{output_dir}/splits/{split_name}/fix/logs", exist_ok=True)
            os.makedirs(f"{output_dir}/splits/{split_name}/fix/results", exist_ok=True)
            with open(f"{output_dir}/splits/{split_name}/fix/results/prompt-{k}.txt", "w") as f:
                f.write(f"Acc: {info[split_name]['acc'] / info[split_name]['cnt'] * 100:.2f}%, Avg length: {info[split_name]['total_length'] / info[split_name]['cnt']:.2f}")
            with open(f"{output_dir}/splits/{split_name}/fix/logs/prompt-{k}.pkl", 'wb') as f:
                assert len(info[split_name]["log"]) == 500
                pickle.dump(info[split_name]["log"], f)
        

if __name__ == "__main__":
    split("meta-llama/Llama-2-7b-chat-hf", "_baselines_correctprompt")
