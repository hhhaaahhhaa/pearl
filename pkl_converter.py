import json
import pickle
from datasets import load_dataset
from tqdm.auto import tqdm
from datamodule import DataModule
from collections import defaultdict

actions_str = [
    'Analyze the given information, break down the problem into manageable steps, apply suitable mathematical operations, and provide a clear, accurate, and concise solution, ensuring precise rounding if necessary. Consider all variables and carefully consider the problem’s context for an efficient solution.',
    'Answer Directly.',
    'Break this down.',
    'Embrace challenges as opportunities for growth. Each obstacle you overcome brings you closer to success.',
    'Let’s be realistic and think step by step.',
    'Let’s solve this problem by splitting it into steps.',
    'Let’s think about this logically.',
    'Let’s think like a detective step by step.',
    'Let’s think step by step.',
    'Let’s work this out in a step by step way to be sure we have the right answer.',
    'Let’s work through this problem step-by-step:',
    'Question decomposition.',
    'Remember that progress is made one step at a time. Stay determined and keep moving forward.',
    'Stay focused and dedicated to your goals. Your consistent efforts will lead to outstanding achievements.',
    'Take a deep breath and work on this problem step-by-step.',
    'Take a deep breath and work on this problem.',
    'Take pride in your work and give it your best. Your commitment to excellence sets you apart.',
    'This is very important to my career.',
    'Write your answer and give me a confidence score between 0-1 for your answer.',
    'You have to solve this problem, I am in trouble.',
    "You'd better be sure."
]

def old_to_new_json(input_dir):
    datamodule_old = DataModule()
    test_dataset_old = datamodule_old.test_dataset
    old_test_dataset_order = ["openbookqa_test", "strategyqa_test", "truthfulqa_test"]
    split_indexes = {
        "openbookqa_test": [0, 500],
        "strategyqa_test": [500, 1000],
        "truthfulqa_test": [1000, 1500]
    }

    dataset = load_dataset("voidful/hint-lm-data")
    
    for split_name in old_test_dataset_order:
        v = dataset[split_name]
        processed_data = []

        for data_item in tqdm(v): 
            data_item['inst'] = defaultdict(dict)
            processed_data.append(data_item)
            
        v = dataset[split_name]
        # for each prompt in prompt pool
        for idx in range(len(actions_str)):
            action_str = actions_str[idx]
            with open(f"{input_dir}/fix/logs/prompt-{idx}.pkl", 'rb') as f:
                # for scores
                scores = pickle.load(f)[split_indexes[split_name][0]:split_indexes[split_name][1]]
                # for response (old) step (new)
                test_dataset_old_split = test_dataset_old[split_indexes[split_name][0]:split_indexes[split_name][1]]
                for data_item, i in tqdm(zip(v, range(len(v)))):
                    text = test_dataset_old_split[i]['prompt'][action_str]
                    score = list(scores[i]['scores'])
                    processed_data[i]['inst'][action_str] = {'step':text, 'score': score}
    
        with open(f'./old/{split_name}_processed_data.json', 'w') as json_file:
            json.dump(processed_data, json_file, indent=4)

if __name__ == "__main__":
    old_to_new_json("_baselines_1218")
