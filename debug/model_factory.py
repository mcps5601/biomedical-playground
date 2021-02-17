import torch
import pytorch_lightning as pl
from transformers import AutoModel, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score

class BertForSTS(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bluebert = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.5)
        self.linear = torch.nn.Linear(self.bluebert.config.hidden_size, 1)
        torch.nn.init.xavier_uniform_(self.linear.weight)  # as weight init of BlueBERT
        self.linear.bias.data.fill_(0.1)  # as bias init of BlueBERT

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_layer = self.bluebert(input_ids, attention_mask, token_type_ids)
        # cls_token = output_layer[0][:, 0, :]
        output_layer = output_layer.pooler_output
        output_layer = self.dropout(output_layer)
        logits = self.linear(output_layer)

        return logits