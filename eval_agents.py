import numpy as np
import random

from llm import LLM
from actor import Actor

system_prompt = "<<SYS>> You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. <</SYS>>"

class BaseAgent(object):

    def eval(self, sample, *args, **kwargs) -> dict:
        # evaluate a sample and return results
        raise NotImplementedError


class NoPromptAgent(BaseAgent):
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def eval(self, sample) -> dict:
        if "llama" in self.llm.model_name or "Llama" in self.llm.model_name:
            final_input = "[INST] " + system_prompt + " " + sample["question"] + " [/INST]"
        else:
            final_input = sample["question"]
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
            response = self.llm(self.llm.format_input_text(sample["question"] + " " + self.prompt), max_new_tokens=10)[0]
        
        if "llama" in self.llm.model_name or "Llama" in self.llm.model_name:
            final_input = "[INST] " + system_prompt + " " + sample["question"] + " " + self.prompt +  "[/INST] " + response + " [INST] Give me your final answer. [/INST]"
        else:
            final_input = sample["question"] + " " + self.prompt + response
        choices_prob = self.llm.score_choice(final_input, sample["options"])
        label = sample["options"].index(sample["answer"])

        return {
            "is_correct": np.argmax(choices_prob).item() == label,
            "response_length": len(self.llm.tokenizer.tokenize(response)),
            "log": {"scores": choices_prob},
        }
    

class RandomPromptAgent(BaseAgent):
    def __init__(self, llm: LLM, prompt_pool: list[str], seed=666) -> None:
        random.seed(seed)
        self.llm = llm
        self.prompt_pool = prompt_pool

    def eval(self, sample) -> dict:
        # get intermediate action
        prompt = random.choice(self.prompt_pool)
        if prompt in sample["prompt"]:  # use precomputed response
            response = sample["prompt"][prompt]
        else:
            response = self.llm(self.llm.format_input_text(sample["question"] + " " + prompt), max_new_tokens=10)[0]
        
        final_input = "[INST] " + system_prompt + " " + sample["question"] + " " + prompt +  "[/INST] " + response + " [INST] Give me your final answer. [/INST]"
        choices_prob = self.llm.score_choice(final_input, sample["options"])
        label = sample["options"].index(sample["answer"])

        return {
            "is_correct": np.argmax(choices_prob).item() == label,
            "response_length": len(self.llm.tokenizer.tokenize(response)),
            "log": {"scores": choices_prob},
        }


class PearlAgent(BaseAgent):  # For future evalutation of trained model
    def __init__(self, llm: LLM, actor: Actor, prompt_pool: list[str]) -> None:
        self.llm = llm
        self.actor = actor
        self.prompt_pool = prompt_pool

    def eval(self, sample) -> dict:
        # get intermediate action
        pred_prompt_id = self.actor.predict(sample["question"])["max_pred"]
        prompt = self.prompt_pool[pred_prompt_id]
        response = sample["prompt"][prompt]
        
        if "llama" in self.llm.model_name or "Llama" in self.llm.model_name:
            final_input = "[INST] " + system_prompt + " " + sample["question"] + " " + prompt +  "[/INST] " + response + " [INST] Give me your final answer. [/INST]"
        else:
            final_input = sample["question"] + " " + prompt + response
        choices_prob = self.llm.score_choice(final_input, sample["options"])
        label = sample["options"].index(sample["answer"])

        return {
            "is_correct": np.argmax(choices_prob).item() == label,
            "response_length": len(self.llm.tokenizer.tokenize(response)),
            "log": {"scores": choices_prob, "prompt_id": pred_prompt_id},
        }
