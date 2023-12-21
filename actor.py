import pfrl
import torch
import torch.nn as nn
from torch import autocast

from Define import PROMPT_POOL


class Actor:
    def __init__(self, env, model, tokenizer,
                 optimizer='sgd',
                 gpu_id=0,
                 act_deterministically=True,
                 temperature=1.0,
                 top_k=0,
                 top_p=1.0):
        self.agent = None
        self.env = env
        self.gpu_id = gpu_id
        self.device = torch.device("cuda:{}".format(gpu_id))
        self.model = model
        if hasattr(model.config, 'word_embed_proj_dim'):
            self.obs_size = model.config.word_embed_proj_dim
        else:
            self.obs_size = model.config.hidden_size
        self.act_deterministically = act_deterministically
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.optimizer = optimizer
        self.converter = torch.nn.Linear(self.obs_size, self.obs_size).to(self.model.dtype)
        parents = [parent[0] for parent in model.named_children()]
        if 'transformer' in parents:  # gpt2/bloom:
            self.transformers_model = model.transformer
        elif 'model' in parents:  # bart
            self.transformers_model = model.model
        elif 'decoder' in parents:  # t5
            self.transformers_model = model.decoder
        else:
            raise ValueError('model not supported')

    def agent_ppo(self, update_interval=10, minibatch_size=3000, epochs=20, lr=3e-6):
        policy = torch.nn.Sequential(
            self.converter,
            pfrl.policies.GaussianHeadWithStateIndependentCovariance(
                action_size=self.obs_size,
                var_type="diagonal",
                var_func=lambda x: torch.exp(2 * x),
                var_param_init=0
            ),
        )
        vf = torch.nn.Sequential(
            torch.nn.Linear(self.obs_size, self.obs_size // 2).to(self.model.dtype),
            torch.nn.Linear(self.obs_size // 2, self.obs_size // 4).to(self.model.dtype),
            torch.nn.Linear(self.obs_size // 4, 2).to(self.model.dtype)
        )
        model = pfrl.nn.Branched(policy, vf)
        model = model.to(dtype=self.env.model_param_dtype)
        if isinstance(self.optimizer, str):
            if self.optimizer.lower() == 'adamw':
                opt = torch.optim.AdamW(model.parameters(), lr=lr)
            else:
                opt = torch.optim.SGD(model.parameters(), lr=lr)
        else:
            opt = self.optimizer
        agent = pfrl.agents.PPO(
            model,
            opt,
            gpu=self.gpu_id,
            update_interval=update_interval,
            minibatch_size=minibatch_size,
            epochs=epochs,
            clip_eps_vf=None,
            entropy_coef=0,
            gamma=0.95,  # https://arxiv.org/abs/2210.01241
            lambd=1,
            max_grad_norm=1.0,
            standardize_advantages=True,
            act_deterministically=self.act_deterministically
        )
        self.agent = agent
        return agent

    # @autocast('cuda')
    def predict(self, input_text):
        with torch.inference_mode():
            with self.agent.eval_mode():
                obs = self.env.reset(input_text)
                action_vec = self.agent.act(obs)
                obs, reward, done, pred_max = self.env.step(action_vec)
                return pred_max



class Adaptor(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.converter = torch.nn.Linear(d_in, d_in)
        self.head = pfrl.policies.SoftmaxCategoricalHead()

    def forward(self, inputs):
        x, feats = inputs
        x = self.converter(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        logits = x.unsqueeze(1) @ feats.transpose(-1, -2)
        logits = logits.squeeze(1)
        # print(logits.shape)
        # input()
        return self.head(logits)


class VF(nn.Module):
    def __init__(self, obs_size):
        super().__init__()
        self.obs_size = obs_size
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.obs_size, self.obs_size // 2),
            torch.nn.Linear(self.obs_size // 2, self.obs_size // 4),
            torch.nn.Linear(self.obs_size // 4, 2)
        )

    def forward(self, inputs):
        x, feats = inputs
        return self.net(x)


class Actor2:
    def __init__(self, env, model, tokenizer,
                 optimizer='sgd',
                 gpu_id=0,
                 act_deterministically=True,
                 temperature=1.0,
                 top_k=0,
                 top_p=1.0):
        self.agent = None
        self.env = env
        self.gpu_id = gpu_id
        self.device = torch.device("cuda:{}".format(gpu_id))
        self.model = model
        if hasattr(model.config, 'word_embed_proj_dim'):
            self.obs_size = model.config.word_embed_proj_dim
        else:
            self.obs_size = model.config.hidden_size
        self.act_deterministically = act_deterministically
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.optimizer = optimizer
        self.adaptor = Adaptor(self.obs_size)
        parents = [parent[0] for parent in model.named_children()]
        if 'transformer' in parents:  # gpt2/bloom:
            self.transformers_model = model.transformer
        elif 'model' in parents:  # bart
            self.transformers_model = model.model
        elif 'decoder' in parents:  # t5
            self.transformers_model = model.decoder
        else:
            raise ValueError('model not supported')

    def agent_ppo(self, update_interval=10, minibatch_size=3000, epochs=20, lr=3e-6):
        vf = VF(self.obs_size)
        model = pfrl.nn.Branched(self.adaptor, vf)
        model = model.to(dtype=self.env.model_param_dtype)
        if isinstance(self.optimizer, str):
            if self.optimizer.lower() == 'adamw':
                opt = torch.optim.AdamW(model.parameters(), lr=lr)
            else:
                opt = torch.optim.SGD(model.parameters(), lr=lr)
        else:
            opt = self.optimizer
        agent = pfrl.agents.PPO(
            model,
            opt,
            gpu=self.gpu_id,
            update_interval=update_interval,
            minibatch_size=minibatch_size,
            epochs=epochs,
            clip_eps_vf=None,
            entropy_coef=0,
            gamma=0.95,  # https://arxiv.org/abs/2210.01241
            lambd=1,
            max_grad_norm=1.0,
            standardize_advantages=True,
            act_deterministically=self.act_deterministically
        )
        self.agent = agent
        return agent

    # @autocast('cuda')
    def predict(self, input_text):
        with torch.inference_mode():
            with self.agent.eval_mode():
                obs = self.env.reset(input_text)
                action = self.agent.act(obs)
                obs, reward, done, pred_max = self.env.step(action)
                return pred_max
