import numpy as np
import torch
from fastchat.model import get_conversation_template
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLM:
    def __init__(self, model_name="gpt2",
                 tokenizer=None,
                 model=None,
                 encoder_decoder=False,
                 use_fastchat_model=False,
                 device="cuda",
                 device_map="auto"):
        if not tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = tokenizer
        self.encoder_decoder = encoder_decoder
        self.device = device
        if not model:
            if encoder_decoder:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16,
                                                                   device_map=device_map)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,
                                                                  device_map=device_map)
        else:
            self.model = model
            self.model = self.model.to(self.device)

        self.model_name = model_name
        self.use_fastchat_model = use_fastchat_model
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'right'

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def input_encode(self, input_sent: str):
        if self.use_fastchat_model:
            conv = get_conversation_template(self.model_path)
            conv.append_message(conv.roles[0], input_sent)
            conv.append_message(conv.roles[1], None)
            input_sent = conv.get_prompt()
        tensor_input = self.tokenizer.encode(input_sent, return_tensors='pt').to(self.device).to(self.model.dtype)
        return tensor_input

    def __call__(self, input_sent: str,
                 do_sample=False,
                 top_k=50,
                 top_p=0.95,
                 typical_p=1.0,
                 no_repeat_ngram_size=0,
                 temperature=1.0,
                 repetition_penalty=1.0,
                 guidance_scale=1,
                 max_new_tokens=512):

        tokenized = self.tokenizer(input_sent, padding=True, return_tensors='pt')
        input_ids = tokenized.input_ids.to(self.device)

        output_ids = self.model.generate(
            input_ids,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            typical_p=typical_p,
            no_repeat_ngram_size=no_repeat_ngram_size,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            guidance_scale=guidance_scale,
            max_new_tokens=max_new_tokens
        )

        actual_seq_lengths = tokenized.attention_mask.sum(dim=1)
        output_ids = [output_id[seq_length:] for output_id, seq_length in zip(output_ids, actual_seq_lengths)]

        predictions = []
        for prediction in self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        ):
            prediction = prediction.strip()
            predictions.append(prediction)
        return predictions

    def score_choice(self, input_sent: str, label_sents: list[str]):
        tensor_input = self.input_encode(input_sent)
        scores = []
        for label_sent in label_sents:
            with torch.inference_mode():
                if self.encoder_decoder:
                    label = self.tokenizer.encode(label_sent, return_tensors='pt').to(self.device).to(self.model.dtype)
                    loss = self.model(input_ids=tensor_input, labels=label).loss
                    scores.append(-loss.item())
                else:
                    input_sent_tokens = self.tokenizer.encode(input_sent, return_tensors='pt',
                                                              add_special_tokens=False).to(self.device)
                    label_sent_tokens = self.tokenizer.encode(label_sent, return_tensors='pt',
                                                              add_special_tokens=False).to(self.device)
                    concatenated = torch.cat(
                        [torch.tensor([[self.tokenizer.eos_token_id]]).to(self.device),
                         input_sent_tokens,
                         label_sent_tokens,
                         torch.tensor([[self.tokenizer.eos_token_id]]).to(self.device)], dim=-1)
                    labels = torch.full_like(concatenated, -100).to(self.device)
                    labels[:, -label_sent_tokens.shape[1] - 1:] = torch.cat(
                        [label_sent_tokens,
                         torch.tensor([[self.tokenizer.eos_token_id]]).to(self.device)],
                        dim=-1)
                    loss = self.model(concatenated, labels=labels).loss.item()
                    average_loss = loss / len(label_sent_tokens[0])
                    scores.append(-average_loss)
        return self.softmax(scores)

    def format_input_text(self, input_text: str) -> str:
        """ Format text for llama2 only! """
        if "llama" in self.model_name or "Llama" in self.model_name:
            system_prompt = "<<SYS>> You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. <</SYS>>"
            return "[INST] " + system_prompt + " " + input_text + " [/INST]"
        else:
            return input_text
