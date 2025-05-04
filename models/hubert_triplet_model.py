from transformers import (
    HubertModel,
    HubertPreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils.generic import ModelOutput

import torch
import torch.nn as nn

from torch.nn import CrossEntropyLoss, TripletMarginLoss

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class HubertTripletModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    embeddings: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class HubertForTripletTrain(HubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.hubert = HubertModel(config)
        self.fc1 = nn.Linear(config.hidden_size, config.classifier_proj_size)

        self.post_init()

    def get_embeddings(self, input_values, attention_mask=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        relu = nn.ReLU(inplace=True)

        hidden_state = outputs.last_hidden_state
        hidden_state = self.fc1(hidden_state.mean(dim=1))
        hidden_state = relu(hidden_state)

        return hidden_state

    def forward(
        self,
        anchor_input_values,
        positive_input_values,
        negative_input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_loss=True,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        anchor_embeddings = self.get_embeddings(anchor_input_values)
        positive_embeddings = self.get_embeddings(positive_input_values)
        negative_embeddings = self.get_embeddings(negative_input_values)

        loss_fn = TripletMarginLoss(margin=1.0)
        loss = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)

        return HubertTripletModelOutput(
            loss=loss,
            embeddings=(anchor_embeddings, positive_embeddings, negative_embeddings),
            attentions=None,
        )


class HubertClassificationAfterTriplet(HubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.hubert = HubertModel(config)
        self.fc1 = nn.Linear(config.hidden_size, config.classifier_proj_size)
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)

        self.post_init()

    def forward(
        self,
        input_values,
        labels=None,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_loss=True,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        relu = nn.ReLU(inplace=True)

        hidden_state = outputs.last_hidden_state
        hidden_state = self.fc1(hidden_state.mean(dim=1))
        hidden_state = relu(hidden_state)
        logits = self.classifier(hidden_state)

        loss = None
        if labels is not None:
            loss_fn = CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.config.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
