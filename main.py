import logging
import sys

from pfrl.experiments import train_agent_with_evaluation

from actor import Actor
from env import Env
from llm import LLM
from datamodule import DataModule

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

import torch

import csv
import wandb
import os

llm = LLM('meta-llama/Llama-2-7b-chat-hf')
# llm = LLM('gpt2')

system_prompt = "<<SYS>> You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. <</SYS>>"

training_logs = []

class MyEnv(Env):

    def reward(self, input_question, prompt_pool_probs):
        # get the max probs id and the corresponding tokenized action
        max_probs_id = torch.argmax(prompt_pool_probs)
        max_prompt_str = self.actions_str[max_probs_id]

        # get intermediate action 
        if max_prompt_str in self.current_sample["prompt"]:  # use precomputed response
            response = self.current_sample["prompt"][max_prompt_str]
        else:
            response = llm("[INST] " + system_prompt + " " + input_question + " " + max_prompt_str + " [/INST]", max_new_tokens=512)[0]
            print("generated")  # Should not happen
        
        # 2nd pass to get score (0-1)
        final_input = "[INST] " + system_prompt + " " + input_question + " " + max_prompt_str + " [/INST] " + response + " [INST] Give me your final answer. [/INST]"
        choices_prob = llm.score_choice(final_input, self.current_output_options)
        
        # score of correct answer divided by intermediate action's length
        # print(choices_prob, self.current_output_options.index(self.current_output), len(response))
        # r = choices_prob[self.current_output_options.index(self.current_output)] / len(response)
        # r = min(r, 10000)

        # reward 1
        # r = choices_prob[self.current_output_options.index(self.current_output)] * 1000 - len(response)

        # reward 2
        # response_token = self.tokenizer(response)["input_ids"]
        # response_token_length = len(response_token)
        # r = choices_prob[self.current_output_options.index(self.current_output)] - response_token_length / 400

        # reward 3
        # r = choices_prob[self.current_output_options.index(self.current_output)]

        response_token = self.tokenizer(response)["input_ids"]
        response_token_length = len(response_token)

        # new_reward_1
        # 1 / response_token_length
        r = 1 / response_token_length

        # current_sample_idx, prompt_id, 
        wandb.log({
            "current_sample_idx": self.current_sample_idx,
            "prompt_id": max_probs_id.item(),
            "response_token_length": response_token_length,
            "reward": r,
            "correct_choice": self.current_output_options.index(self.current_output),
            "choices_prob": choices_prob,
            })
        
        training_logs.append([
            self.current_sample_idx,
            max_probs_id.item(),
            response_token_length,
            r,
            self.current_output_options.index(self.current_output),
            *choices_prob
            ])

        return r


def main():
    tokenizer = llm.tokenizer
    model = llm.model
    env = MyEnv(model, tokenizer, datamodule=DataModule())
    actor = Actor(env, model, tokenizer)
    agent = actor.agent_ppo(update_interval=10, minibatch_size=1, epochs=20)

    steps=40000
    eval_n_steps=None
    eval_n_episodes=300
    train_max_episode_len=50
    eval_interval=2000
    outdir='new_reward_1_3'

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
    with open(os.path.join(outdir, "log.txt"), "w") as fout:
        writer = csv.writer(fout)

        writer.writerows(training_logs)

if __name__ == "__main__":
    main()
