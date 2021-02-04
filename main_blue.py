import argparse
import os, sys
sys.path.append('bluebert')

from transformers import AutoModel, AutoTokenizer, get_polynomial_decay_schedule_with_warmup
from blue_factory import BiossesDataset
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils import set_seed
from scipy.stats import pearsonr, spearmanr


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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=True)

    train_data = BiossesDataset(tokenizer, os.path.join(args.data_dir, args.data_name, 'train.tsv'),
                                args.max_seq_len)
    trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    dev_data = BiossesDataset(tokenizer, os.path.join(args.data_dir, args.data_name, 'dev.tsv'),
                              args.max_seq_len)
    devloader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    test_data = BiossesDataset(tokenizer, os.path.join(args.data_dir, args.data_name, 'test.tsv'),
                               args.max_seq_len)
    testloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    model = BertForBLUE(args)
    model = model.to(device)
    model.train()

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-06)

    # learning rate scheduler (linear)
    num_training_steps = int(len(trainloader) * args.epochs)
    num_warmup_steps = int(num_training_steps * args.warmup_proportion)

    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps,
                                                lr_end=0.0,
                                                power=1.0,
                                                last_epoch=-1)  #cycle=False

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
            loss = loss_fn(outputs.squeeze(-1), scores)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Update learning rate schedule
            scheduler.step()

            train_loss += loss.item()

        print("Epoch {}, train_loss: {}".format(epoch, train_loss/len(trainloader)))

        if epoch  == args.epochs:
            # start evaluating in each epoch
            model.eval()
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
                    loss = loss_fn(outputs.squeeze(-1), scores)
                    dev_loss += loss.item()
                    pearson = pearsonr(outputs.squeeze(-1).cpu().numpy(), scores.cpu().numpy())[0]

            print("Epoch {}, valid_loss: {}, pearson:{} ".format(epoch, dev_loss/len(devloader), pearson))

            

    
    
    
    
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
        default=3,
        type=int
    )
    parser.add_argument(
        '--criteria',
        default='mse',
        type=str
    )
    parser.add_argument(
        '--learning_rate',
        default=5e-5,
        type=float
    )
    parser.add_argument(
        '--max_seq_len',
        default=128,
        type=int
    )
    parser.add_argument(
        '--warmup_proportion',
        default=0.1,
        type=float
    )
    parser.add_argument(
        '--weight_decay',
        default=0.01,
        type=float
    )
    # parser.add_argument(
    #     '--accumulation',
    #     default=2,
    #     type=int
    # )
    # parser.add_argument(
    #     '--bert_config_file',
        
    # )
    args = parser.parse_args()

    main(args)
