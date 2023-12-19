import torch
import torch.nn as nn

from llm import LLM


class MeanPooling(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, feature_BxTxH, features_len, **kwargs):
        ''' 
        Arguments
            feature_BxTxH - [BxTxH]   Acoustic feature with shape 
            features_len  - [B] of feature length
        '''
        agg_vec_list = []
        for i in range(len(feature_BxTxH)):
            agg_vec = torch.mean(feature_BxTxH[i][:features_len[i]], dim=0)
            agg_vec_list.append(agg_vec)

        return torch.stack(agg_vec_list)  # B, H


class LinearProbe(nn.Module):
    def __init__(self, llm: LLM, n_cls: int) -> None:
        super().__init__()
        self.llm = llm
        self.model = llm.model
        self.tokenizer = llm.tokenizer
        # llm d_hidden
        self.obs_size = 2560  # fixed to 2560 only for phi2 model
        # if hasattr(self.model.config, 'word_embed_proj_dim'):
        #     self.obs_size = self.model.config.word_embed_proj_dim
        # else:
        #     self.obs_size = self.model.config.hidden_size

        self.head = nn.Sequential(
            nn.Linear(self.obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.pooling = MeanPooling()
        self.head2 = nn.Linear(128, n_cls)

    def input_template(self, sample) -> str:
        """ Can modify this template """
        return sample["question"]

    def forward(self, samples):
        outs = []
        for sample in samples:
            x = self.input_template(sample)
            # print(x)
            # input()
            with torch.no_grad():
                out = self._get_observation(x)
                out = out.to(torch.float32)
                outs.append(out)
        lengths = [len(out) for out in outs]
        # print(lengths)
        total_length = max(lengths)
        x = nn.utils.rnn.pad_sequence(outs, batch_first=True)
        x = self.pooling(x, lengths)
        x = self.head2(self.head(x))
        # print(outs.shape)
        # input()

        return x

    def _get_observation(self, input_text):
        feature_dict = self.tokenizer(input_text,
                                      return_tensors='pt',
                                      return_token_type_ids=False,
                                      add_special_tokens=False).to(self.model.device)
        # with torch.cuda.amp.autocast(enabled=False):
        prediction = self.model(**feature_dict, output_hidden_states=True)
        # print(prediction.hidden_states.shape)
        # input()
        outputs = prediction.hidden_states[:, :, :]
        return outputs.data[-1]

    def build_optimized_model(self) -> nn.Module:
        optimized_modules = nn.ModuleList([self.head, self.head2])
        cnt = sum([p.numel() for p in optimized_modules.parameters() if p.requires_grad])
        print(f"Optimizable: {cnt}")
        return optimized_modules


# Old, we'll use linear layer only for simplicity
# class LSTMProbe(nn.Module):
#     def __init__(self, llm: LLM, n_cls: int) -> None:
#         super().__init__()
#         self.llm = llm
#         self.model = llm.model
#         self.tokenizer = llm.tokenizer
#         # llm d_hidden
#         if hasattr(self.model.config, 'word_embed_proj_dim'):
#             self.obs_size = self.model.config.word_embed_proj_dim
#         else:
#             self.obs_size = self.model.config.hidden_size

#         self.lstm = nn.LSTM(input_size=self.obs_size, hidden_size=128, 
#                                 num_layers=2, bidirectional=True, batch_first=True)
#         self.pooling = MeanPooling()
#         self.head = nn.Linear(256, n_cls)

#     def forward(self, samples):
#         outs = []
#         for sample in samples:
#             x = sample["question"]
#             with torch.no_grad():
#                 out = self._get_observation(x)
#                 outs.append(out)
#         lengths = [len(out) for out in outs]
#         total_length = max(lengths)
#         x = nn.utils.rnn.pad_sequence(outs, batch_first=True)
#         # print(lengths)
#         # print(x.shape)
#         x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
#         x, _ = self.lstm(x)  # B, L, d_out
#         x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=total_length)
#         # print(x.shape)
#         x = self.pooling(x, lengths)
#         x = self.head(x)
#         # print(x.shape)
#         # input()

#         return x

#     def _get_observation(self, input_text):
#         feature_dict = self.tokenizer(input_text,
#                                       return_tensors='pt',
#                                       return_token_type_ids=False,
#                                       add_special_tokens=False).to(self.model.device)
#         # with torch.cuda.amp.autocast(enabled=False):
#         prediction = self.model(**feature_dict, output_hidden_states=True)
#         outputs = prediction.hidden_states[-1][:, :, :]
#         return outputs.data[-1]

#     def build_optimized_model(self) -> nn.Module:
#         optimized_modules = nn.ModuleList([self.lstm, self.head])
#         cnt = sum([p.numel() for p in optimized_modules.parameters() if p.requires_grad])
#         print(f"Optimizable: {cnt}")
#         return optimized_modules
