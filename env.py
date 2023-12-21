import gym
import random

import numpy
import torch
from torch import autocast

from Define import PROMPT_POOL


class Env(gym.Env):
    def __init__(self, model, tokenizer, datalist):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.observation_input = datalist
        self.hidden_size = model.get_output_embeddings().weight.shape[-1]
        self.action_space = gym.spaces.Discrete(self.hidden_size)
        self.model_param_dtype = next(model.parameters()).dtype
        self.current_input = ""
        self.current_output = ""
        self.current_output_options = []
        self.current_sample = None

        self.actions_str = [
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
            "You'd better be sure."]
        self.tokenized_actions = [self.tokenizer.tokenize(i) for i in self.actions_str]

        self.reset()

    def step(self, observation_vector):
        predictions, done, max_pred = self._predict(observation_vector)
        reward = self.reward(self.current_input, predictions)
        return self.get_observation(self.current_input), reward, done, {"max_pred": max_pred}

    def reward(self, input_question, prompt_pool_probs):
        return 0

    def reset(self, input_text=None):
        if input_text:  # not used
            self.current_input = input_text
            self.current_output = "dummy1"
            self.current_output_options = ["dummy1", "dummy2"]
        else:
            sample = random.choice(self.observation_input)
            self.current_sample = sample
            self.current_input = sample["question"]
            self.current_output = sample["answer"]
            self.current_output_options = sample["options"]
        return self.get_observation(self.current_input)

    @autocast('cuda')
    def get_observation(self, input_text):
        feature_dict = self.tokenizer(f"For the question, '{input_text}', identify the most effective method to infer the answer, taking efficiency into consideration.",
                                      return_tensors='pt',
                                      return_token_type_ids=False,
                                      add_special_tokens=False).to(self.model.device)
        # with torch.cuda.amp.autocast(enabled=False):
        prediction = self.model(**feature_dict, output_hidden_states=True)
        outputs = prediction.hidden_states[:, -1, :]
        return outputs.data[-1]

    def extract_sent_embedding(self, sent_vectors):
        return sent_vectors[-1]

    def _predict(self, observation_vector):
        if isinstance(observation_vector, numpy.ndarray):
            observation_vector = torch.from_numpy(observation_vector).to(self.model.device)
        self.extract_sent_embedding(observation_vector)
        feats = torch.stack(
            [self.get_observation(f"Instruct: With Consider that {i}: {self.current_input}.") for i in self.actions_str])
        feats = feats / feats.norm(dim=1, keepdim=True)
        logits = observation_vector @ feats.t()
        return logits, True, logits.argmax(dim=-1).item()



class Env2(gym.Env):
    def __init__(self, model, tokenizer, datalist):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.observation_input = datalist
        # self.hidden_size = model.get_output_embeddings().weight.shape[-1]
        self.action_space = gym.spaces.Discrete(len(PROMPT_POOL))
        self.model_param_dtype = next(model.parameters()).dtype
        self.current_input = ""
        self.current_output = ""
        self.current_output_options = []
        self.current_sample = None

        self.actions_str = PROMPT_POOL
        self.tokenized_actions = [self.tokenizer.tokenize(i) for i in self.actions_str]

        self.reset()

    def step(self, action):
        reward = self.reward(self.current_input, action)
        return (self.get_observation(self.current_input), self.feats), reward, True, {"max_pred": action}

    def reward(self, input_question, prompt_pool_probs):
        return 0

    def reset(self, input_text=None):
        if input_text:  # not used
            self.current_input = input_text
            self.current_output = "dummy1"
            self.current_output_options = ["dummy1", "dummy2"]
        else:
            sample = random.choice(self.observation_input)
            self.current_sample = sample
            self.current_input = sample["question"]
            self.current_output = sample["answer"]
            self.current_output_options = sample["options"]
        
        self.feats = torch.stack(
            [self.get_observation(f"Instruct: With Consider that {i}: {self.current_input}.") for i in self.actions_str])
        
        return (self.get_observation(self.current_input), self.feats)

    @autocast('cuda')
    def get_observation(self, input_text):
        feature_dict = self.tokenizer(f"For the question, '{input_text}', identify the most effective method to infer the answer, taking efficiency into consideration.",
                                      return_tensors='pt',
                                      return_token_type_ids=False,
                                      add_special_tokens=False).to(self.model.device)
        # with torch.cuda.amp.autocast(enabled=False):
        prediction = self.model(**feature_dict, output_hidden_states=True)
        outputs = prediction.hidden_states[:, -1, :]
        return outputs.data[-1]
