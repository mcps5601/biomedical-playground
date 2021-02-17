import torch
import pytorch_lightning as pl
from transformers import AutoModel, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score

class BertForSTS(torch.nn.Module):
    def __init__(self, args, checkpoint=None):
        super().__init__()
        self.args = args
        self.bluebert = AutoModel.from_pretrained(self.args.model_name)
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

class BertForNLI(pl.LightningModule):
    def __init__(self, args, checkpoint=None):
        super().__init__()
        self.args = args
        self.model = AutoModelForSequenceClassification.from_pretrained(self.args.model_name)

    def training_step(self, batch, batch_idx):
        text, segments, attention_masks, labels = batch
        outputs = model(input_ids=text, 
                        token_type_ids=segments, 
                        attention_mask=attention_masks,
                        labels=labels)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        text, segments, attention_masks, labels = batch
        outputs = model(input_ids=text, 
                        token_type_ids=segments, 
                        attention_mask=attention_masks,
                        labels=labels)
        _, y_hat = torch.max(outputs.logits, dim=1)
        val_acc = accuracy_score(y_hat.cpu(), labels.cpu())
        return {'val_loss': outputs.loss, 'val_acc': val_acc}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
