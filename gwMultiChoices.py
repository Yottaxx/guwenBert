from abc import ABC

import torch.nn as nn
from torch.cuda.amp import autocast
from transformers import BertModel, ElectraModel, AutoModel, AlbertModel,RobertaModel
import torch


class gwBert(nn.Module, ABC):

    def __init__(self, model, d_in=768, d_out=1, dropout=0.1):
        super().__init__()
        self.model = AutoModel.from_pretrained(model)
        self.classifier = nn.Linear(d_in, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with autocast():
            choices = input_ids.shape[1]
            dim = input_ids.shape[-1]
            # print("------------")
            # print(input_ids[:2])
            input_ids = input_ids.reshape(-1, dim)
            # print(input_ids[:4])
            attention_mask = attention_mask.reshape(-1, dim)
            # token_type_ids = token_type_ids.reshape(-1, dim)

            inputs = dict()
            inputs['input_ids'] = input_ids
            inputs['attention_mask'] = attention_mask
            x = self.model(**inputs)[1]
            # pooled_out = x[0][:,0,:]
            # print(x.shape)
            pooled_out = x
            # print(x[0].shape)
            pooled_out = self.dropout(pooled_out)
            logits = self.classifier(pooled_out).squeeze()
            logits = logits.view(-1, choices)
        return logits
