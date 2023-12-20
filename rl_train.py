import nlp2
import math
import logging
import sys

from pfrl.experiments import train_agent_with_evaluation

from actor import Actor
from env import Env
from llm import LLM

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

import torch

import csv
import wandb
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sup.hack import PhiForCausalLM
from llm import LLM
torch.set_default_device("cuda")
model = PhiForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", flash_attn=True, flash_rotary=True, fused_dense=True, device_map="cuda", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer.pad_token_id=tokenizer.eos_token_id
llm = LLM(model=model,tokenizer=tokenizer)
training_logs = []

def calculate_entropy(probabilities):
    entropy = 0
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log(p)
    return entropy


data_list = []
for s in [
    "hotpotqa_train",
    "openbookqa_train",
    "strategyqa_train",
    "truthfulqa_train",
    "hotpotqa_validation",
    "openbookqa_test",
    "strategyqa_test",
    "truthfulqa_test",
]:
    data_list.extend(nlp2.read_json(f'{s}_processed_data.json'))
q_key_pool = {}
for i in data_list:
    q_key_pool[i['question']] = {'inst':i['inst'],'opt':i['options'],'ans':i['answer'],'ans_idx':i['options'].index(i['answer'])}
     
class MyEnv(Env):
    def reward(self, input_question, prompt_pool_probs):
        global q_key_pool
        max_probs_id = torch.argmax(prompt_pool_probs)
        max_prompt_str = self.actions_str[max_probs_id]
        q_data = q_key_pool[input_question]
        choices_prob = q_data['inst'][max_prompt_str]

        # To modify
        r = 1/calculate_entropy(choices_prob['score']) * 1/len(tokenizer.tokenize(choices_prob['step'])) * 10
        
        wandb.log({
            "prompt_id": max_probs_id,
            "response_token_length": len(tokenizer.tokenize(choices_prob['step'])),
            "reward": r,
            "correct_choice": q_data['ans_idx'],
            "choices_prob": choices_prob,
        })
        
        return r

tokenizer = llm.tokenizer
model = llm.model
env = MyEnv(model, tokenizer, datalist=data_list)
actor = Actor(env, model, tokenizer,optimizer='sgd')
agent = actor.agent_ppo(update_interval=50, minibatch_size=5, epochs=20, lr=5e-5)

steps=30000
eval_n_steps=None
eval_n_episodes=300
train_max_episode_len=50
eval_interval=1000
outdir='reward_v2'

wandb.init(
    project="RL_final_training",
    name=outdir,

    config={
        "steps": steps,
        "eval_n_steps": eval_n_steps,
        "eval_n_episodes": eval_n_episodes,
        "train_max_episode_len": train_max_episode_len,
        "eval_interval": eval_interval,
        "outdir": outdir,
    }
)

train_agent_with_evaluation(
    agent,
    env,
    steps=steps,
    eval_n_steps=eval_n_steps,
    eval_n_episodes=eval_n_episodes,
    train_max_episode_len=train_max_episode_len,
    eval_interval=eval_interval,
    outdir=outdir,
)
wandb.finish()