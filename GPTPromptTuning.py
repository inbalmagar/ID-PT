import os
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Model
import torch
import torch.nn as nn


class GPTPromptTuning:
    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: str,
            n_tokens: int = 20,
            initialize_from_vocab: bool = True,
            random_range: float = 0.5,
            **kwargs,
    ):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Make sure to freeze Transformers model
        for param in model.parameters():
            param.requires_grad = False

        print("Initializing soft prompt...")
        model.initialize_soft_prompt(
            n_tokens=n_tokens,
            initialize_from_vocab=initialize_from_vocab,
            random_range=random_range,
        )

        return model

    def initialize_soft_prompt(self,
                               n_tokens: int = 20,
                               initialize_from_vocab: bool = True,
                               random_range: float = 0.5,
    ) -> None:
        self.n_tokens = n_tokens
        if initialize_from_vocab:
            init_prompt_value = self.transformer.wte.weight[:n_tokens].clone().detach()

        else:
            init_prompt_value = torch.FloatTensor(2, 10).uniform_(-random_range, random_range)
        self.soft_prompt = nn.Embedding(n_tokens, self.config.n_embd)
        # Initialize weight
        self.soft_prompt.weight = nn.parameter.Parameter(init_prompt_value)

    def save_soft_prompt(self, path: str, filename: str = "soft_prompt.model"):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.soft_prompt, os.path.join(path, filename))

    def _cat_learned_embedding_to_input(self, input_ids) -> torch.Tensor:
        inputs_embeds = self.transformer.wte(input_ids)

        # for single input (without batch)
        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        # [batch_size, n_tokens, n_embd]
        learned_embeds = self.soft_prompt.weight.repeat(inputs_embeds.size(0), 1, 1)
        inputs_embeds = torch.cat([learned_embeds, inputs_embeds], dim=1)
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

    def forward(
            self,
            input_ids=None,
            inputs_embeds=None,
            attention_mask=None,
            labels=None,
            return_dict=None,
    ):
        if input_ids is not None:
            inputs_embeds = self._cat_learned_embedding_to_input(input_ids).to(self.device)

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


class GPTPromptTuningNoLabels:
    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path: str,
                        n_tokens: int = 20,
                        initialize_from_vocab: bool = True,
                        random_range: float = 0.5,
                        **kwargs,
                        ):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Make sure to freeze Transformers model
        for param in model.parameters():
            param.requires_grad = False

        print("Initializing soft prompt...")
        model.initialize_soft_prompt(
            n_tokens=n_tokens,
            initialize_from_vocab=initialize_from_vocab,
            random_range=random_range,
        )

        return model

    def initialize_soft_prompt(self,
                               n_tokens: int = 20,
                               initialize_from_vocab: bool = True,
                               random_range: float = 0.5,
    ) -> None:
        self.n_tokens = n_tokens
        if initialize_from_vocab:
            init_prompt_value = self.wte.weight[:n_tokens].clone().detach()

        else:
            init_prompt_value = torch.FloatTensor(2, 10).uniform_(-random_range, random_range)
        self.soft_prompt = nn.Embedding(n_tokens, self.config.n_embd)
        # Initialize weight
        self.soft_prompt.weight = nn.parameter.Parameter(init_prompt_value)

    def save_soft_prompt(self, path: str, filename: str = "soft_prompt.model"):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.soft_prompt, os.path.join(path, filename))

    def _cat_learned_embedding_to_input(self, input_ids) -> torch.Tensor:
        inputs_embeds = self.wte(input_ids)

        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        # [batch_size, n_tokens, n_embd]
        learned_embeds = self.soft_prompt.weight.repeat(inputs_embeds.size(0), 1, 1)

        inputs_embeds = torch.cat([learned_embeds, inputs_embeds], dim=1)

        return inputs_embeds

    # def _extend_labels(self, labels, ignore_index=-100) -> torch.Tensor:
    #     if len(list(labels.shape)) == 1:
    #         labels = labels.unsqueeze(0)
    #
    #     n_batches = labels.shape[0]
    #     return torch.cat(
    #         [
    #             torch.full((n_batches, self.n_tokens), ignore_index).to(self.device),
    #             labels,
    #         ],
    #         dim=1,
    #     )

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
                return_dict=None,
    ):
        if input_ids is not None:
            inputs_embeds = self._cat_learned_embedding_to_input(input_ids).to(
                self.device
            )

        # if labels is not None:
        #     labels = self._extend_labels(labels).to(self.device)

        if attention_mask is not None:
            attention_mask = self._extend_attention_mask(attention_mask).to(self.device)

        return super().forward(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            # labels=labels,
            return_dict=return_dict,
        )


class GPT2PromptTuningModel(GPTPromptTuningNoLabels, GPT2Model):
    def __init__(self, config):
        super().__init__(config)


class GPT2PromptTuningLM(GPTPromptTuning, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
