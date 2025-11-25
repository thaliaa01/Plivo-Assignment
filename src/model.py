import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from labels import LABEL2ID, ID2LABEL

class PIIModel(nn.Module):
    def __init__(self, base_model="distilbert-base-cased"):
        super().__init__()
        cfg = AutoConfig.from_pretrained(base_model)
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = cfg.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, len(LABEL2ID)),
        )
        self.id2label = ID2LABEL
        self.label2id = LABEL2ID

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq = self.dropout(output.last_hidden_state)
        logits = self.classifier(seq)
        return logits
