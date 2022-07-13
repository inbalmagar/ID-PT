import os
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np
from GPTPromptTuning import GPT2PromptTuningModel


class GPTIDPromptTuning:
    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path: str,
                        n_tokens: int = 20,
                        initialize_from_vocab: bool = True,
                        random_range: float = 0.5,
                        **kwargs, ):

        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Make sure to freeze Tranformers model
        for param in model.parameters():
            param.requires_grad = False

        # get the frozen model for ID-PT
        model.get_prompt_model(pretrained_model_name_or_path,
                               n_tokens,
                               initialize_from_vocab,
                               random_range)

        return model

    def get_prompt_model(self,
                         pretrained_model_name_or_path,
                         n_tokens,
                         initialize_from_vocab,
                         random_range):
        self.n_tokens = n_tokens

        self.prompt_model = GPT2PromptTuningModel.from_pretrained(pretrained_model_name_or_path,
                                                                  n_tokens=n_tokens,
                                                                  initialize_from_vocab=initialize_from_vocab,
                                                                  random_range=random_range)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", pad_token='<|pad|>')
        self.prompt_model.resize_token_embeddings(len(tokenizer))
        self.prompt_model.config.pad_token_id = tokenizer.pad_token_id

    def save_soft_prompt(self, path: str, filename: str = "soft_prompt.model"):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.soft_prompt, os.path.join(path, filename))

    def _generate_id_prompt(self, input_ids, attention_mask):
        output = self.prompt_model(input_ids, attention_mask)

        hidden = output['last_hidden_state']
        sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
        sequence_range = torch.LongTensor(
            np.array([torch.arange(x - self.n_tokens + 1, x + 1).numpy() for x in sequence_lengths]))
        sequence_range = sequence_range.unsqueeze(-1)
        indices = sequence_range.repeat(1, 1, self.config.n_embd).to(self.device)
        pooled_hidden = torch.gather(hidden, 1, indices)

        return self._cat_generated_embedding_to_input(input_ids, pooled_hidden).to(self.device)

    def _cat_generated_embedding_to_input(self, input_ids, prompt) -> torch.Tensor:
        inputs_embeds = self.transformer.wte(input_ids)

        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        inputs_embeds = torch.cat([prompt, inputs_embeds], dim=1)

        return inputs_embeds

    def _extend_labels(self, labels, ignore_index=-100) -> torch.Tensor:
        if len(list(labels.shape)) == 1:
            labels = labels.unsqueeze(0)

        n_batches = labels.shape[0]
        return torch.cat(
            [
                torch.full((n_batches, self.n_tokens), ignore_index).to(self.device),
                labels,
            ],
            dim=1,
        )

    def _extend_attention_mask(self, attention_mask):

        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)

        n_batches = attention_mask.shape[0]
        return torch.cat(
            [torch.full((n_batches, self.n_tokens), 1).to(self.device), attention_mask],
            dim=1,
        )

    def forward(self,
                input_ids=None,
                inputs_embeds=None,
                attention_mask=None,
                labels=None,
                return_dict=None):
        if input_ids is not None:
            inputs_embeds = self._generate_id_prompt(input_ids, attention_mask).to(self.device)

        if attention_mask is not None:
            attention_mask = self._extend_attention_mask(attention_mask).to(self.device)

        if labels is not None:
            labels = self._extend_labels(labels).to(self.device)

        return super().forward(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            return_dict=return_dict,
        )


class GPT2IDPromptTuningLM(GPTIDPromptTuning, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
