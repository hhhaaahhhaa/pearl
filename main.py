import logging
import sys

from pfrl.experiments import train_agent_with_evaluation

from actor import Actor
from env import Env
from llm import LLM

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "facebook/opt-125m"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")
model = model.cuda()


class MyEnv(Env):
    def reward(self, input_question, prompt_pool_probs):
        # get the max probs id and the corresponding tokenized action
        max_probs_id = torch.argmax(prompt_pool_probs)
        max_prompt_tokenized = self.tokenized_actions[max_probs_id]
        max_prompt_str = self.actions_str[max_probs_id]
        llm = LLM(model=self.model)
        response = llm(input_question + " " + max_prompt_str, max_new_tokens=400)[0]
        choices_prob = llm.score_choice(input_question + " " + max_prompt_str + response, ["True", "False"])
        print(input_question + " " + max_prompt_str)
        print(response)
        print(choices_prob)
        # if true is correct answer, and its id is 0
        return choices_prob[0] / len(max_prompt_tokenized)


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
