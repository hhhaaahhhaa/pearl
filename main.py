import logging
import sys

from pfrl.experiments import train_agent_with_evaluation

from actor import Actor
from env import Env
from llm import LLM
from datamodule import DataModule

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

import torch

llm = LLM('meta-llama/Llama-2-7b-chat-hf')
# llm = LLM('gpt2')


class MyEnv(Env):

    def reward(self, input_question, prompt_pool_probs):
        # get the max probs id and the corresponding tokenized action
        max_probs_id = torch.argmax(prompt_pool_probs)
        max_prompt_str = self.actions_str[max_probs_id]

        # TODO: need debug
        # get intermediate action 
        if max_prompt_str in self.current_sample["prompt"]:  # use precomputed response
            response = self.current_sample["prompt"][max_prompt_str]
            print("precomputed")
        else:
            response = llm(input_question + " " + max_prompt_str, max_new_tokens=10)[0]
            print("generated")
        input()
        
        # 2nd pass to get score (0-1)
        choices_prob = llm.score_choice(input_question + " " + max_prompt_str + response, self.current_output_options)
        
        # score of correct answer divided by intermediate action's length
        r = choices_prob[self.current_output_options.index(self.current_output)] / len(max_prompt_str)

        return r


def main():
    tokenizer = llm.tokenizer
    model = llm.model
    env = MyEnv(model, tokenizer, datamodule=DataModule())
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


if __name__ == "__main__":
    main()
