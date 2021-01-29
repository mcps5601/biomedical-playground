import argparse
import os, sys
sys.path.append('bluebert')

from transformers import AutoModel, AutoTokenizer
from blue_factory import BiossesDataset
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils import set_seed


def collate_fn(batch):
    text = [i[0] for i in batch]
    text = pad_sequence(text, batch_first=True)
    segments = [i[1] for i in batch]
    segments = pad_sequence(segments, batch_first=True)
    score = torch.stack([i[2] for i in batch])
    masks_tensors = torch.zeros(text.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(text != 0, 1)

    return text, segments, masks_tensors, score


class BertForBLUE(torch.nn.Module):
    def __init__(self, args, checkpoint=None):
        super().__init__()
        self.args = args
        self.bluebert = AutoModel.from_pretrained(self.args.model_name)
        self.dropout = torch.nn.Dropout(0.5)
        self.linear = torch.nn.Linear(self.bluebert.config.hidden_size, 1)
        torch.nn.init.xavier_uniform_(self.linear.weight)  # as weight init of BlueBERT
        self.linear.bias.data.fill_(0.1)  # as bias init of BlueBERT

    def forward(self, input_ids, token_type_ids, attention_mask):
        output_layer = self.bluebert(input_ids, token_type_ids, attention_mask)
        # cls_token = output_layer[0][:, 0, :]
        output_layer = output_layer.pooler_output
        output_layer = self.dropout(output_layer)
        logits = self.linear(output_layer)

        return logits


def main(args):
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_data = BiossesDataset(tokenizer, os.path.join(args.data_dir, args.data_name, 'train.tsv'), args.max_seq_len)
    for i in train_data:
        if len(i[0]) > 128:
            print(i)
    exit()
    trainloader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=collate_fn)

    dev_data = BiossesDataset(tokenizer, os.path.join(args.data_dir, args.data_name, 'dev.tsv'), args.max_seq_len)
    devloader = DataLoader(dev_data, batch_size=args.batch_size, collate_fn=collate_fn)

    test_data = BiossesDataset(tokenizer, os.path.join(args.data_dir, args.data_name, 'test.tsv'), args.max_seq_len)
    testloader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    model = BertForBLUE(args)
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.criteria == 'mse':
        loss_fn = torch.nn.MSELoss()

    for epoch in range(1, args.epochs+1):
        # start training in each epoch
        train_loss = 0
        for train_batch in trainloader:
            optimizer.zero_grad()

            text = train_batch[0].to(device)
            segments = train_batch[1].to(device)
            attention_masks = train_batch[2].to(device)
            scores = train_batch[3].to(device)

            outputs = model(input_ids=text, 
                            token_type_ids=segments, 
                            attention_mask=attention_masks)
            loss = loss_fn(outputs, scores)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        print("Epoch {}, train_loss: {}".format(epoch, train_loss/len(trainloader)))
        
        if epoch % 5 == 0:
            # start evaluating in each epoch
            print("=========================================")
            dev_loss = 0
            for dev_batch in devloader:
                text = dev_batch[0].to(device)
                segments = dev_batch[1].to(device)
                attention_masks = dev_batch[2].to(device)
                scores = dev_batch[3].to(device)

                with torch.no_grad():
                    outputs = model(input_ids=text,
                                    token_type_ids=segments,
                                    attention_mask=attention_masks)
                    loss = loss_fn(outputs, scores)
                    dev_loss += loss.item()
        
            print("Epoch {}, valid_loss: {}".format(epoch, dev_loss/len(devloader)))

            

    
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name',
        default='bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12',
        type=str
    )
    parser.add_argument(
        '--task_name',
        default='sts',
        choices=['sts', 'nli'],
        type=str
    )
    parser.add_argument(
        '--data_name',
        default='BIOSSES',
        choices=['BIOSSES', 'MEDSTS'],
        type=str
    )
    parser.add_argument(
        '--data_dir',
        default='/home/dean/datasets/benchmarks/BLUE/data_v0.2/data/',
        type=str
    )
    parser.add_argument(
        '--seed',
        default=777,
        type=int
    )
    parser.add_argument(
        '--batch_size',
        default=32,
        type=int
    )
    parser.add_argument(
        '--epochs',
        default=100,
        type=int
    )
    parser.add_argument(
        '--criteria',
        default='mse',
        type=str
    )
    parser.add_argument(
        '--learning_rate',
        default=2e-4,
        type=float
    )
    parser.add_argument(
        '--max_seq_len',
        default=128,
        type=int
    )
    # parser.add_argument(
    #     '--bert_config_file',
        
    # )
    args = parser.parse_args()

    main(args)
