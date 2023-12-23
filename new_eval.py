import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import nlp2
from llm import LLM
from actor import Actor, Actor2
from env import Env, Env2
torch.set_default_device("cuda")

from Define import TEST_SPLIT_NAMES, PROMPT_POOL
from new_eval_agents import *


def main(input_dir, output_dir):
    # load model
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", flash_attn=True, flash_rotary=True, fused_dense=True, device_map="cuda", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    tokenizer.pad_token_id=tokenizer.eos_token_id
    llm = LLM(model=model,tokenizer=tokenizer)
    for s in TEST_SPLIT_NAMES:
        evaluator = FixPromptAgent(llm, cache=f"{input_dir}/{s}_processed_data.json")
        res = evaluator.get_info(["acc", "length", "entropy"])
        os.makedirs(f'{output_dir}/fix', exist_ok=True)
        with open(f'{output_dir}/fix/{s}.json', 'w') as f:
            json.dump(res, f, indent=4)

        evaluator = RandomAgent(llm, cache=f"{input_dir}/{s}_processed_data.json")
        res = evaluator.get_info(["acc", "length", "entropy"])
        os.makedirs(f'{output_dir}/random', exist_ok=True)
        with open(f'{output_dir}/random/{s}.json', 'w') as f:
            json.dump(res, f, indent=4)

        evaluator = ToplineAgent(llm, cache=f"{input_dir}/{s}_processed_data.json")
        res = evaluator.get_info(["acc", "length", "entropy"])
        os.makedirs(f'{output_dir}/topline', exist_ok=True)
        with open(f'{output_dir}/topline/{s}.json', 'w') as f:
            json.dump(res, f, indent=4)


def main_sup(input_dir, output_dir, ckpt_path: str):
    from sup_baseline import PhiForCausalLM, LinearProbe
    # load model
    phi_model = PhiForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", flash_attn=True, flash_rotary=True, fused_dense=True, device_map="cuda", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    tokenizer.pad_token_id=tokenizer.eos_token_id
    llm = LLM(model=phi_model,tokenizer=tokenizer)
    model = LinearProbe(llm=llm, n_cls=len(PROMPT_POOL))
    model.cuda()
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    
    for s in TEST_SPLIT_NAMES:
        evaluator = SupAgent(model, cache=f"{input_dir}/{s}_processed_data.json")
        res = evaluator.get_info(["acc", "length", "entropy", "predictions"])
        os.makedirs(output_dir, exist_ok=True)
        with open(f'{output_dir}/{s}.json', 'w') as f:
            json.dump(res, f, indent=4)


def main_rl(input_dir, output_dir, checkpoint_dir: str):
    from sup_baseline import PhiForCausalLM

    # load LLM
    phi_model = PhiForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", flash_attn=True, flash_rotary=True, fused_dense=True, device_map="cuda", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    tokenizer.pad_token_id=tokenizer.eos_token_id
    llm = LLM(model=phi_model,tokenizer=tokenizer)

    # load actor
    tokenizer = llm.tokenizer
    model = llm.model

    # use baseclass since inference will not use the reward function, and put any dummy json here simply for initialization
    env = Env(model, tokenizer, datalist=nlp2.read_json(f"{input_dir}/hotpotqa_validation_processed_data.json"))
    actor = Actor(env, model, tokenizer)
    # env = Env2(model, tokenizer, datalist=nlp2.read_json(f"{input_dir}/hotpotqa_validation_processed_data.json"))
    # actor = Actor2(env, model, tokenizer)
    agent = actor.agent_ppo(update_interval=100, minibatch_size=5, epochs=20)
    print(f"Load from {checkpoint_dir}...")
    agent.load(checkpoint_dir)

    for s in TEST_SPLIT_NAMES:
        evaluator = PearlAgent(llm, actor, cache=f"{input_dir}/{s}_processed_data.json")
        res = evaluator.get_info(["acc", "length", "entropy", "predictions"])
        os.makedirs(output_dir, exist_ok=True)
        with open(f'{output_dir}/{s}.json', 'w') as f:
            json.dump(res, f, indent=4)


if __name__ == "__main__":
    # main("_data", "_data/analysis")
    # main_sup("_data", "_data/analysis/sup-debug", "sup-debug/checkpoints/3.ckpt")
    main_rl("_data", "_data/analysis/reward_v5", "reward_v5/best")
    # main_rl("_data", "_data/analysis/rlfinal-debug", "rlfinal-debug/best")
