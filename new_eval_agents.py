import numpy as np
import random
import json
from tqdm import tqdm

from llm import LLM
from actor import Actor
from Define import PROMPT_POOL


class BaseAgent(object):

    def eval(self, sample, *args, **kwargs) -> dict:
        # evaluate a sample and return results
        raise NotImplementedError
    

class FixPromptAgent(BaseAgent):
    def __init__(self, llm: LLM, cache: str):
        self.llm = llm
        with open(cache, 'r') as f:
            self.info = json.load(f)

    def get_info(self, keys):
        res = {p: {} for p in PROMPT_POOL}
        if 'acc' in keys:
            for p in PROMPT_POOL:
                res[p]['acc'] = self.get_acc(p)
        if 'length' in keys:
            for p in PROMPT_POOL:
                res[p]['length'] = self.get_length(p)
        if 'entropy' in keys:
            for p in PROMPT_POOL:
                res[p]['entropy'] = self.get_entropy(p)
        return res

    def get_acc(self, prompt: str):
        acc = 0
        for sample in self.info:
            label = sample["options"].index(sample["answer"])
            choices_prob = sample['inst'][prompt]['score']
            is_correct = (np.argmax(choices_prob).item() == label)
            acc += is_correct
        return acc / len(self.info)

    def get_length(self, prompt: str):
        length = 0
        for sample in self.info:
            response = sample['inst'][prompt]['step']
            response_length = len(self.llm.tokenizer.tokenize(response))
            length += response_length
        return length / len(self.info)
    
    def get_entropy(self, prompt: str):
        entropy = 0
        for sample in self.info:
            choices_prob = sample['inst'][prompt]['score']
            entropy += float(-np.sum(choices_prob * np.log(choices_prob)))
        return entropy / len(self.info)

    

class FlexiblePromptAgent(BaseAgent):
    def __init__(self, llm: LLM, cache: str):
        self.llm = llm
        with open(cache, 'r') as f:
            self.info = json.load(f)

    def prompt_selection(self, sample) -> str:
        raise NotImplementedError
    
    def get_info(self, keys):
        res = {}
        if 'acc' in keys:
            res['acc'] = self.get_acc()
        if 'length' in keys:
            res['length'] = self.get_length()
        if 'entropy' in keys:
            res['entropy'] = self.get_entropy()
        return res
    
    def get_acc(self):
        acc = 0
        for sample in self.info:
            prompt = self.prompt_selection(sample)
            label = sample["options"].index(sample["answer"])
            choices_prob = sample['inst'][prompt]['score']
            is_correct = (np.argmax(choices_prob).item() == label)
            acc += is_correct
        return acc / len(self.info)

    def get_length(self):
        length = 0
        for sample in self.info:
            prompt = self.prompt_selection(sample)
            response = sample['inst'][prompt]['step']
            response_length = len(self.llm.tokenizer.tokenize(response))
            length += response_length
        return length / len(self.info)
    
    def get_entropy(self):
        entropy = 0
        for sample in self.info:
            prompt = self.prompt_selection(sample)
            choices_prob = sample['inst'][prompt]['score']
            entropy += float(-np.sum(choices_prob * np.log(choices_prob)))
        return entropy / len(self.info)


class RandomAgent(FlexiblePromptAgent):
    def __init__(self, llm: LLM, cache: str):
        super().__init__(llm, cache)

    def prompt_selection(self, sample) -> str:
        return random.choice(PROMPT_POOL)


class ToplineAgent(FlexiblePromptAgent):
    def __init__(self, llm: LLM, cache: str):
        super().__init__(llm, cache)

    def prompt_selection(self, sample) -> str:
        best_prompt, best_length = None, 1e9
        for prompt in PROMPT_POOL:
            choices_prob = sample['inst'][prompt]['score']
            label = sample["options"].index(sample["answer"])
            if np.argmax(choices_prob).item() == label:
                if best_prompt is None:
                    best_prompt = prompt
                    response = sample['inst'][prompt]['step']
                    response_length = len(self.llm.tokenizer.tokenize(response))
                    best_length = response_length
                else:
                    response = sample['inst'][prompt]['step']
                    response_length = len(self.llm.tokenizer.tokenize(response))
                    if best_length > response_length:
                        best_prompt = prompt
                        best_length = response_length
        if best_prompt is None:
            return random.choice(PROMPT_POOL)
        return best_prompt


class SupAgent(FlexiblePromptAgent):
    def __init__(self, model, cache: str):
        self.model = model
        self.llm = self.model.llm
        with open(cache, 'r') as f:
            self.info = json.load(f)

    def infer_all(self):
        self.predictions = {}
        for sample in tqdm(self.info):
            logits = self.model([sample])
            pred_prompt_id = logits[0].argmax().item()
            self.predictions[sample['question']] = pred_prompt_id
            
    def get_info(self, keys):
        self.infer_all()
        res = {}
        if 'acc' in keys:
            res['acc'] = self.get_acc()
        if 'length' in keys:
            res['length'] = self.get_length()
        if 'entropy' in keys:
            res['entropy'] = self.get_entropy()
        if 'predictions' in keys:
            res['predictions'] = self.predictions
        return res
    
    def prompt_selection(self, sample) -> str:
        pred_prompt_id = self.predictions[sample['question']]
        return PROMPT_POOL[pred_prompt_id]


class PearlAgent(FlexiblePromptAgent):  # For future evalutation of trained model
    def __init__(self, llm: LLM, actor: Actor, cache: str) -> None:
        super().__init__(llm, cache)
        self.actor = actor

    def infer_all(self):
        self.predictions = {}
        for sample in tqdm(self.info):
            pred_prompt_id = self.actor.predict(sample["question"])["max_pred"]
            self.predictions[sample['question']] = pred_prompt_id
            
    def get_info(self, keys):
        self.infer_all()
        res = {}
        if 'acc' in keys:
            res['acc'] = self.get_acc()
        if 'length' in keys:
            res['length'] = self.get_length()
        if 'entropy' in keys:
            res['entropy'] = self.get_entropy()
        if 'predictions' in keys:
            res['predictions'] = self.predictions
        return res
    
    def prompt_selection(self, sample) -> str:
        pred_prompt_id = self.predictions[sample['question']]
        return PROMPT_POOL[pred_prompt_id]
