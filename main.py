import logging
import sys

from pfrl.experiments import train_agent_with_evaluation

from actor import Actor
from env import Env

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "facebook/opt-125m"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")
model = model.cuda()


class MyEnv(Env):
    def reward(self, input_item, predicted_list, finish):
        reward = 0
        return reward


env = MyEnv(model, tokenizer, observation_input=["hello world", "how are you", "I am fine"])
actor = Actor(env, model, tokenizer)
agent = actor.agent_ppo(update_interval=10, minibatch_size=2000, epochs=20)

train_agent_with_evaluation(
    agent,
    env,
    steps=1000,
    eval_n_steps=None,
    eval_n_episodes=1500,
    train_max_episode_len=50,
    eval_interval=10000,
    outdir='somewhere',
)
