import gym
import random

import numpy
import torch
from torch import autocast


class Env(gym.Env):
    def __init__(self, model, tokenizer, observation_input=[]):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.observation_input = observation_input
        self.hidden_size = model.get_output_embeddings().weight.shape[-1]
        self.action_space = gym.spaces.Discrete(self.hidden_size)
        self.model_param_dtype = next(model.parameters()).dtype
        self.current_input = ""

        self.actions_str = ["Let’s work this out in a step by step way to be sure we have the right answer",
                            "Retrieve information from search engine",
                            "Break this down into smaller steps",
                            "Take a deep breath and work on this problem step-by-step",
                            "A little bit of arithmetic and a logical approach will help us quickly arrive the solution to this problem",
                            "Question decomposition",
                            "Answer Directly"]
        self.tokenized_actions = [self.tokenizer.tokenize(i) for i in self.actions_str]

        self.reset()

    def step(self, observation_vector):
        predictions, done, max_pred = self._predict(observation_vector)
        reward = self.reward(self.current_input, predictions, done)
        return self.get_observation(self.current_input), reward, done, {"max_pred": max_pred}

    def reward(self, input_item, predicted_list, done):
        return 0

    def reset(self, input_text=None):
        if input_text:
            self.current_input = input_text
        else:
            self.current_input = random.choice(self.observation_input) if self.observation_input else ""
        return self.get_observation(self.current_input)

    @autocast('cuda')
    def get_observation(self, input_text):
        feature_dict = self.tokenizer(input_text,
                                      return_tensors='pt',
                                      return_token_type_ids=False,
                                      add_special_tokens=False).to(self.model.device)
        # with torch.cuda.amp.autocast(enabled=False):
        prediction = self.model(**feature_dict, output_hidden_states=True)
        outputs = prediction.hidden_states[-1][:, -1, :]
        return outputs.data[-1]

    def extract_sent_embedding(self, sent_vectors):
        return sent_vectors[-1]

    def _predict(self, observation_vector):
        if isinstance(observation_vector, numpy.ndarray):
            observation_vector = torch.from_numpy(observation_vector).to(self.model.device)
        self.extract_sent_embedding(observation_vector)
        feats = torch.stack(
            [self.get_observation(self.current_input + i) for i in self.actions_str])
        feats = feats / feats.norm(dim=1, keepdim=True)
        logits = observation_vector @ feats.t()
        return logits, True, logits.argmax(dim=-1).item()
