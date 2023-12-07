import numpy as np
import random

from llm import LLM


class BaseAgent(object):

    def eval(self, sample, *args, **kwargs) -> dict:
        # evaluate a sample and return results
        raise NotImplementedError


class NoPromptAgent(BaseAgent):
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def eval(self, sample) -> dict:
        choices_prob = self.llm.score_choice(sample["question"], sample["options"])
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
        
        final_input = sample["question"] + " " + self.prompt + response
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
        
        final_input = sample["question"] + " " + prompt + response
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
