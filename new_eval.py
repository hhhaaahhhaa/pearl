import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
from llm import LLM
torch.set_default_device("cuda")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", flash_attn=True, flash_rotary=True, fused_dense=True, device_map="cuda", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer.pad_token_id=tokenizer.eos_token_id
llm = LLM(model=model,tokenizer=tokenizer)


from Define import TEST_SPLIT_NAMES, PROMPT_POOL
from new_eval_agents import *


def main():
    for s in TEST_SPLIT_NAMES:
        evaluator = FixPromptAgent(llm, cache=f"_data/{s}_processed_data.json")
        res = evaluator.get_info(["acc", "length", "entropy"])
        os.makedirs(f'_data/analysis/fix', exist_ok=True)
        with open(f'_data/analysis/fix/{s}.json', 'w') as f:
            json.dump(res, f, indent=4)

        evaluator = RandomAgent(llm, cache=f"_data/{s}_processed_data.json")
        res = evaluator.get_info(["acc", "length", "entropy"])
        os.makedirs(f'_data/analysis/random', exist_ok=True)
        with open(f'_data/analysis/random/{s}.json', 'w') as f:
            json.dump(res, f, indent=4)

        evaluator = ToplineAgent(llm, cache=f"_data/{s}_processed_data.json")
        res = evaluator.get_info(["acc", "length", "entropy"])
        os.makedirs(f'_data/analysis/topline', exist_ok=True)
        with open(f'_data/analysis/topline/{s}.json', 'w') as f:
            json.dump(res, f, indent=4)


def main_sup():
    for s in TEST_SPLIT_NAMES:
        evaluator = SupAgent(model, cache=f"_data/{s}_processed_data.json")
        res = evaluator.get_info(["acc", "length", "entropy", "predictions"])
        os.makedirs(f'_data/analysis/sup', exist_ok=True)
        with open(f'_data/analysis/sup/{s}.json', 'w') as f:
            json.dump(res, f, indent=4)


if __name__ == "__main__":
    main()
    main_sup()
