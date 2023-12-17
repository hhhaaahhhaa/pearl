import numpy as np
import random

from llm import LLM

system_prompt = "<<SYS>> You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. <</SYS>>"

class BaseAgent(object):

    def eval(self, sample, *args, **kwargs) -> dict:
        # evaluate a sample and return results
        raise NotImplementedError


class NoPromptAgent(BaseAgent):
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def eval(self, sample) -> dict:
        final_input = "[INST] " + system_prompt + " " + sample["question"] + " [/INST] [INST] Give me your final answer. [/INST]"
        # choices_prob = self.llm.score_choice(sample["question"], sample["options"])
        choices_prob = self.llm.score_choice(final_input, sample["options"])
        label = sample["options"].index(sample["answer"])

        return {
            "is_correct": np.argmax(choices_prob).item() == label,
            "response_length": 0,
            "log": {"scores": choices_prob},
        }


class FixPromptAgent(BaseAgent):
    def __init__(self, llm: LLM, prompt: str) -> None:
        self.llm = llm
        self.prompt = prompt

    def eval(self, sample) -> dict:
        # get intermediate action
        if self.prompt in sample["prompt"]:  # use precomputed response
            response = sample["prompt"][self.prompt]
        else:
            response = self.llm(sample["question"] + " " + self.prompt, max_new_tokens=10)[0]
        
        # final_input = sample["question"] + " " + self.prompt + response
        final_input = "[INST] " + system_prompt + " " + sample["question"] + " [/INST] " + self.prompt + " " + response + " [INST] Give me your final answer. [/INST]"
        choices_prob = self.llm.score_choice(final_input, sample["options"])
        label = sample["options"].index(sample["answer"])

        return {
            "is_correct": np.argmax(choices_prob).item() == label,
            "response_length": len(response),
            "log": {"scores": choices_prob},
        }
    

class RandomPromptAgent(BaseAgent):
    def __init__(self, llm, prompt_pool: list[str], seed=666) -> None:
        random.seed(seed)
        self.llm = llm
        self.prompt_pool = prompt_pool

    def eval(self, sample) -> dict:
        # get intermediate action
        prompt = random.choice(self.prompt_pool)
        if prompt in sample["prompt"]:  # use precomputed response
            response = sample["prompt"][prompt]
        else:
            response = self.llm(sample["question"] + " " + prompt, max_new_tokens=10)[0]
        
        # final_input = sample["question"] + " " + prompt + response
        final_input = "[INST] " + system_prompt + " " + sample["question"] + " [/INST] " + prompt + " " + response + " [INST] Give me your final answer. [/INST]"
        choices_prob = self.llm.score_choice(final_input, sample["options"])
        label = sample["options"].index(sample["answer"])

        return {
            "is_correct": np.argmax(choices_prob).item() == label,
            "response_length": len(response),
            "log": {"scores": choices_prob},
        }


class PearlAgent(BaseAgent):  # For future evalutation of trained model
    def __init__(self) -> None:
        pass

    def eval(self, sample) -> dict:
        pass
