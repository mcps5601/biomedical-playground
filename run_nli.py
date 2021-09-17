import argparse
import os, sys
sys.path.append('bluebert')

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_polynomial_decay_schedule_with_warmup
)
from model_factory import BertForSTS
from data_factory import BiossesDataset, MednliDataset
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from utils import set_seed
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score


def collate_fn(batch):
    text = [i[0] for i in batch]
    text = pad_sequence(text, batch_first=True)
    segments = [i[1] for i in batch]
    segments = pad_sequence(segments, batch_first=True)
    label = torch.stack([i[2] for i in batch])
    # print(label.type())
    # exit()
    masks_tensors = torch.zeros(text.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(text != 0, 1)

    return text, segments, masks_tensors, label


def main(args):
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=True)
    processors = {
        "sts": BiossesDataset,
        "nli": MednliDataset,
    }
    processor = processors[args.task_name]

    train_data = processor(tokenizer, os.path.join(args.data_dir, args.data_name, 'train.tsv'),
                                args.max_seq_len)
    trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    dev_data = processor(tokenizer, os.path.join(args.data_dir, args.data_name, 'dev.tsv'),
                              args.max_seq_len)
    devloader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    test_data = processor(tokenizer, os.path.join(args.data_dir, args.data_name, 'test.tsv'),
                               args.max_seq_len)
    testloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.task_name == 'sts':
        model = BertForSTS(args)
    elif args.task_name == 'nli':
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=3)
        for name, param in model.named_parameters():
            if 'classifier.weight' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'classifier.bias' in name:
                param.data.fill_(0)

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
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.task_name))
    for epoch in range(1, args.epochs+1):
        # start training in each epoch
        train_loss = 0
        for train_batch in trainloader:
            optimizer.zero_grad()

            text = train_batch[0].to(device)
            segments = train_batch[1].to(device)
            attention_masks = train_batch[2].to(device)
            labels = train_batch[3].to(device)

            outputs = model(input_ids=text, 
                            token_type_ids=segments, 
                            attention_mask=attention_masks,
                            labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Update learning rate schedule
            scheduler.step()

            train_loss += loss.item()
            writer.add_scalar('train_loss/{}'.format(args.exp_name),
                              train_loss,
                              epoch)

        print("Epoch {}, train_loss: {}".format(epoch, train_loss/len(trainloader)))

        if epoch % 2 == 0:
            # start evaluating
            model.eval()
            print("=========================================")
            dev_loss = 0
            for dev_batch in devloader:
                text = dev_batch[0].to(device)
                segments = dev_batch[1].to(device)
                attention_masks = dev_batch[2].to(device)
                labels = dev_batch[3].to(device)

                with torch.no_grad():
                    outputs = model(input_ids=text,
                                    token_type_ids=segments,
                                    attention_mask=attention_masks,
                                    labels=labels)
                    loss = outputs.loss
                    dev_loss += loss.item()
                    logits = outputs.logits
                    _, y_hat = torch.max(logits, dim=1)
                    val_acc = accuracy_score(y_hat.cpu(), labels.cpu())

            print("Epoch {}, valid_loss: {}, valid_acc:{} ".format(
                                                        epoch,
                                                        dev_loss/len(devloader),
                                                        val_acc))
        if epoch  == args.epochs:
            # start testing
            print("=========================================")
            model.eval()
            test_loss = 0
            for test_batch in testloader:
                text = test_batch[0].to(device)
                segments = test_batch[1].to(device)
                attention_masks = test_batch[2].to(device)
                labels = test_batch[3].to(device)

                with torch.no_grad():
                    outputs = model(input_ids=text,
                                    token_type_ids=segments,
                                    attention_mask=attention_masks,
                                    labels=labels)
                    loss = outputs.loss
                    test_loss += loss.item()
                    logits = outputs.logits
                    _, y_hat = torch.max(logits, dim=1)
                    test_acc = accuracy_score(y_hat.cpu(), labels.cpu())

            print("Epoch {}, test_loss: {}, test_acc:{} ".format(
                                                        epoch,
                                                        test_loss/len(testloader),
                                                        test_acc))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp_name',
        default='lr_1e-5',
        type=str
    )
    parser.add_argument(
        '--model_name',
        default='bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12',
        type=str
    )
    parser.add_argument(
        '--task_name',
        default='nli',
        choices=['sts', 'nli'],
        type=str
    )
    parser.add_argument(
        '--data_name',
        default='mednli',
        choices=['BIOSSES', 'MEDSTS', 'mednli'],
        type=str
    )
    parser.add_argument(
        '--data_dir',
        default='/home/dean/datasets/benchmarks/BLUE/data_v0.2/data/',
        type=str
    )
    parser.add_argument(
        '--log_dir',
        default='./logs',
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
        default=2,
        type=int
    )
    parser.add_argument(
        '--criteria',
        default='mse',
        type=str
    )
    parser.add_argument(
        '--learning_rate',
        default=1e-5,
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
