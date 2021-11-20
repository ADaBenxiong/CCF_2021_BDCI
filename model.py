import os
import sys
import numpy as np
import json

import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers import BertPreTrainedModel, BertModel, PretrainedConfig, PreTrainedModel

class QuestionMatching(BertPreTrainedModel):
    def __init__(self, config, rdrop_coef = 0.0):
        super().__init__(config)

        self.rdrop_coef = rdrop_coef
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.hidden_size_last3 = config.hidden_size * 3
        self.dropout_prob = config.hidden_dropout_prob
        config.output_hidden_states = True
        self.bert = BertModel(config)

        self.dropout = nn.Dropout(self.dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)
        self.last_classifier = nn.Linear(self.hidden_size_last3, self.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids = None,       #g:输入维度 (batch_size, max_seq_length)
            attention_mask = None,
            token_type_ids = None,
            labels = None,          #g:输入维度 (batch_size, label)
    ):

        outputs = self.bert(
            input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
        )

        sequence_output = outputs[0]    
        pooled_output = outputs[1]      
        hidden_output = outputs[2]      

        last_cat = torch.cat((pooled_output, hidden_output[-1][:, 0], hidden_output[-2][:, 0]), 1)
        last_output_linear = self.dropout(last_cat)
        logits = self.last_classifier(last_output_linear)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels) #g: logits的维度是（[batch_size, 2]）, labels的维度是([batch_size, 1])

        output = (logits, ) + outputs[2:]

        return ((loss,) + output) if loss is not None else output

class QuestionMatchingLast3EmbeddingCls(BertPreTrainedModel):
    def __init__(self, config, rdrop_coef = 0.0):
        super().__init__(config)

        self.rdrop_coef = rdrop_coef
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.hidden_size_last4 = config.hidden_size * 4
        self.dropout_prob = config.hidden_dropout_prob
        config.output_hidden_states = True
        self.bert = BertModel(config)

        self.dropout = nn.Dropout(self.dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)
        self.last_classifier = nn.Linear(self.hidden_size_last4, self.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids = None,       #g:输入维度(batch_size, max_seq_length)
            attention_mask = None,
            token_type_ids = None,
            labels = None,          #g:输出维度 (batch_size, label)
    ):

        outputs = self.bert(
            input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
        )

        sequence_output = outputs[0]    
        pooled_output = outputs[1]      
        hidden_output = outputs[2]  

        last_cat = torch.cat((pooled_output, hidden_output[-1][:, 0], hidden_output[-2][:, 0], hidden_output[-3][:, 0]), 1)
        last_output_linear = self.dropout(last_cat)
        logits = self.last_classifier(last_output_linear)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        output = (logits, ) + outputs[2:]

        return ((loss,) + output) if loss is not None else output


















