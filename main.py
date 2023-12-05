import logging
import sys

from pfrl.experiments import train_agent_with_evaluation

from actor import Actor
from env import Env
from llm import LLM

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

import torch

llm = LLM('meta-llama/Llama-2-7b-chat-hf')
tokenizer = llm.tokenizer
model = llm.model


class MyEnv(Env):

    def reward(self, input_question, prompt_pool_probs):
        # get the max probs id and the corresponding tokenized action
        max_probs_id = torch.argmax(prompt_pool_probs)
        max_prompt_str = self.actions_str[max_probs_id]
        response = llm(input_question + " " + max_prompt_str, max_new_tokens=10)[0]
        choices_prob = llm.score_choice(input_question + " " + max_prompt_str + response, ["penguin", "polar bear"])
        return choices_prob[0]


env = MyEnv(model, tokenizer, observation_input=["Select the animal is living in Antarctica?"])
actor = Actor(env, model, tokenizer)
agent = actor.agent_ppo(update_interval=10, minibatch_size=1, epochs=20)

train_agent_with_evaluation(
    agent,
    env,
    steps=1000,
    eval_n_steps=None,
    eval_n_episodes=300,
    train_max_episode_len=50,
    eval_interval=10000,
    outdir='somewhere',
)
