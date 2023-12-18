from datasets import load_dataset


dataset = load_dataset("voidful/hint-lm-data")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm import LLM
torch.set_default_device("cuda")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", flash_attn=True, flash_rotary=True, fused_dense=True, device_map="cuda", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer.pad_token_id=tokenizer.eos_token_id
llm = LLM(model=model,tokenizer=tokenizer)

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

from collections import defaultdict
from tqdm.auto import tqdm
import json

for k, v in dataset.items():
    processed_data = []

    if "full" in k:
        continue
    if "train" not in k:
        continue
    if "truthfulqa" not in k:
        continue
    print(k)
    for data_item in tqdm(v): 
        q = data_item['question']
        if 'truthfulqa' in k or 'hotpotqa' in k:
            ans_prompt = f'Answer the question with yes or no. Question: {q}\nOutput:'
        if 'strategyqa' in k:
            ans_prompt = f'Answer the question with True or False. Question: {q}\nOutput:'
        if 'openbookqa' in k:
            ops_string = "\n".join(data_item['options'])
            ans_prompt = f'Answer the question with Options:\n{ops_string}. Question: {q}\nOutput:'
        data_item['inst'] = defaultdict(dict)

        for i in actions_str:
            prompt = f"Instruct: {i} For the Question: {q}\nOutput:"
            inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
            output_ids = model.generate(**inputs, max_length=512, eos_token_id=tokenizer.eos_token_id)
            actual_seq_lengths = inputs.attention_mask.sum(dim=1)
            output_ids = [output_id[seq_length:] for output_id, seq_length in zip(output_ids, actual_seq_lengths)]
            text = tokenizer.batch_decode(output_ids, skip_special_tokens=True,)[0]
            prompt = f'Instruct: With Consider that {i}: {text}. {ans_prompt}'
            scores = llm.score_choice(prompt,data_item['options'])
            data_item['inst'][i] = {'step':text,'score':list(scores)}
        print(data_item)
        processed_data.append(data_item)
    
    with open(f'./final_data/{k}_processed_data.json', 'w') as json_file:
        json.dump(processed_data, json_file, indent=4)
