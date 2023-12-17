from datasets import load_dataset
from tqdm import tqdm


class DataModule(object):
    def __init__(self):
        self.ds = load_dataset("kuanhuggingface/hint-lm-data")
        self.train_split_names = [
            "hotpotqa_train",
            "hotpotqa_validation",
            "openbookqa_train",
            "openbookqa_validation",
            "strategyqa_train",
            "truthfulqa_train",
        ]
        self.test_split_names = [
            "openbookqa_test",
            "strategyqa_test",
            "truthfulqa_test",
        ]

        self.train_dataset, self.test_dataset = None, None
        self._init_datasets()

    def _init_datasets(self):
        self.train_dataset = []
        for split_name in self.train_split_names:
            for instance in tqdm(self.ds[split_name]):
                sample = {
                    "split_name": split_name,
                    "question": instance["question"],
                    "options": instance["options"],
                    "answer": instance["answer"],
                    "prompt": instance["prompt"],
                }
                self.train_dataset.append(sample)
        
        self.test_dataset = []
        for split_name in self.test_split_names:
            for instance in tqdm(self.ds[split_name]):
                sample = {
                    "split_name": split_name,
                    "question": instance["question"],
                    "options": instance["options"],
                    "answer": instance["answer"],
                    "prompt": instance["prompt"],
                }
                self.test_dataset.append(sample)
