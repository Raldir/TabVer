import json
import math
import os
from statistics import mean

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn

from src.constants import (
    ANSWER_CHOICES_IDS,
    CHOICES_SCORES,
    CHOICES_SCORES_LIST,
    CLAIM_ID,
    CLAIM_SPAN_POS,
    IDX,
    INPUT_IDS,
    LABEL,
    LAST_SPAN,
    LM_TARGET,
    NEG_INDEX,
    OP,
    PRED_PROB,
    PRED_PROB_LIST,
    PREDICTION,
    TEMPLATE_LOGITS,
)
from src.utils.get_optimizer import get_optimizer
from src.utils.get_scheduler import get_scheduler


class DummyLayer(nn.Module):
    """
    DummyLayer to ensure that the gradient checkpointing will assign output layer as require_grad=True.
    Reference: https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
    """

    def __init__(self):
        super().__init__()
        self.dummy_bias = torch.ones(1, dtype=torch.float32, requires_grad=True)

    def forward(self, x):
        return x + self.dummy_bias.to(x) - self.dummy_bias.to(x)


class EncoderDecoder(nn.Module):
    """
    Encoder Decoder
    """

    def __init__(self, config, tokenizer, transformer):
        """
        :param config
        """
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.model = transformer
        self.encoder = (
            self.model.encoder
            if "BART0" not in config.origin_model and "bart" not in config.origin_model
            else self.model.get_encoder()
        )
        self.padding_token = -100 if "BART0" not in config.origin_model and "bart" not in config.origin_model else -1
        self.softmax = nn.Softmax(dim=-1)
        self.max_answer_choice_length = config.max_answer_choice_length

        self.gradient_checkpointing = config.gradient_checkpointing
        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.dummy_layer = DummyLayer()

    def _run_model(self, input_ids, choices_ids, bs, num_choices):
        flat_choices_ids = choices_ids.flatten(0, 1)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()  # [bs, max_seq_len]

        inputs_embeds = self.encoder.embed_tokens(input_ids)

        if self.gradient_checkpointing:
            inputs_embeds = self.dummy_layer(inputs_embeds)

        encoder_hidden_states = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)[0]
        encoder_hidden_states = encoder_hidden_states.unsqueeze(dim=1).repeat(1, num_choices, 1, 1).flatten(0, 1)
        attention_mask = attention_mask.unsqueeze(dim=1).repeat(1, num_choices, 1).flatten(0, 1)
        decoder_input_ids = torch.cat([torch.zeros_like(flat_choices_ids[:, :1]), flat_choices_ids[:, :-1]], dim=1)
        decoder_attention_mask = (decoder_input_ids == decoder_input_ids).float()
        lm_target = flat_choices_ids + self.padding_token * (flat_choices_ids == self.tokenizer.pad_token_id).long()

        model_output = self.model(
            attention_mask=attention_mask,
            encoder_outputs=[encoder_hidden_states],
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False,
        )
        choices_scores = (
            F.cross_entropy(model_output.logits.flatten(0, 1), lm_target.flatten(0, 1), reduction="none")
            .view(bs, num_choices, -1)
            .sum(dim=-1)
        )
        if self.config.length_norm > 0:
            choices_scores = choices_scores / torch.pow(
                (choices_ids != self.tokenizer.pad_token_id).sum(dim=-1), self.config.length_norm
            )
        choices_scores = -choices_scores

        return choices_scores, lm_target, model_output.logits

    def forward(self, batch):
        """
        Predict the lbl for particular pet
        :param batch:
        :param pet:
        :return:
        """
        input_ids, choices_ids, labels = batch[INPUT_IDS], batch[ANSWER_CHOICES_IDS], batch[LABEL]

        # When running inference allow to split the choices to enable evaluation on datasets with many classes
        if self.training:
            bs, num_choices = choices_ids.size()[:2]
            choices_scores, lm_target, model_output = self._run_model(input_ids, choices_ids, bs, num_choices)
            pred_score, prediction = choices_scores.max(dim=1)
        else:
            bs, num_choices = choices_ids.size()[:2]
            splits = math.ceil(num_choices / self.config.split_choices_at)

            choices_ids_split = torch.split(choices_ids, self.config.split_choices_at, dim=1)

            all_choice_scores = []

            for half_choice_ids in choices_ids_split:
                half_num_choices = half_choice_ids.shape[1]

                choices_scores, lm_target, model_output = self._run_model(
                    input_ids, half_choice_ids, bs, half_num_choices
                )
                all_choice_scores.append(choices_scores)

            choices_scores = torch.cat(all_choice_scores, dim=-1)

            choices_scores = self.softmax(choices_scores)
            pred_score, prediction = choices_scores.max(dim=1)

        batch_output = {
            PREDICTION: prediction.tolist(),
            LABEL: labels.tolist(),
            IDX: batch[IDX].tolist(),
            CLAIM_ID: batch[CLAIM_ID].tolist(),
            CLAIM_SPAN_POS: batch[CLAIM_SPAN_POS].tolist(),
            OP: batch[OP].tolist(),
            PRED_PROB_LIST: pred_score.tolist(),
            NEG_INDEX: batch[NEG_INDEX].tolist(),
            LAST_SPAN: batch[LAST_SPAN].tolist(),
            CHOICES_SCORES_LIST: choices_scores.tolist(),
        }
        if self.training:
            train_v = {
                CHOICES_SCORES: choices_scores,
                LM_TARGET: lm_target,
                TEMPLATE_LOGITS: model_output,
            }
            batch_output.update(train_v)
        return batch_output
