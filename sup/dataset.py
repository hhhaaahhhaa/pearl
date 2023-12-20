import numpy as np
import torch
from torch.utils.data import Dataset
import json
import random

from Define import PROMPT_POOL


class LabelDataset(Dataset):
    def __init__(self, data_paths, tokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.data = []
        for path in data_paths:
            with open(path, 'r') as f:
                self.data.extend(json.load(f))
        random.seed(666)

        self.filter_data()

    def filter_data(self):
        res = []
        for sample in self.data:
            if self.prompt_selection(sample) is not None:
                res.append(sample)
        self.data = res
    
    def __getitem__(self, index):
        best_prompt = self.prompt_selection(self.data[index])
        if best_prompt is None:
            best_prompt = random.choice(PROMPT_POOL)

        return {
            "sample": self.data[index],
            "label": PROMPT_POOL.index(best_prompt)
        }

    def __len__(self):
        return len(self.data)
    
    def prompt_selection(self, sample) -> str:
        best_prompt, best_length = None, 1e9
        for prompt in PROMPT_POOL:
            choices_prob = sample['inst'][prompt]['score']
            label = sample["options"].index(sample["answer"])
            if np.argmax(choices_prob).item() == label:
                if best_prompt is None:
                    best_prompt = prompt
                    response = sample['inst'][prompt]['step']
                    response_length = len(self.tokenizer.tokenize(response))
                    best_length = response_length
                else:
                    response = sample['inst'][prompt]['step']
                    response_length = len(self.tokenizer.tokenize(response))
                    if best_length > response_length:
                        best_prompt = prompt
                        best_length = response_length
        return best_prompt

    def collate_fn(self):
        def f(data):
            res = {}
            data_size = len(data)
            idx_arr = np.arange(data_size)
            res["samples"] = [data[idx]["sample"] for idx in idx_arr]
            res["labels"] = torch.LongTensor([data[idx]["label"] for idx in idx_arr])
            return res
        
        return f
