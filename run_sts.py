import argparse
import os, sys
sys.path.append('bluebert')

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_polynomial_decay_schedule_with_warmup,
    get_linear_schedule_with_warmup
)
from model_factory import BertForSTS
from data_factory import BiossesDataset, MednliDataset
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
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


def main(args):
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        do_lower_case=True
    )
    processors = {
        "sts": BiossesDataset,
        "nli": MednliDataset,
    }
    processor = processors[args.task_name]

    train_data = processor(
        tokenizer,
        os.path.join(args.data_dir, args.data_name, 'train.tsv'),
        args.max_seq_len
    )
    trainloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    dev_data = processor(
        tokenizer,
        os.path.join(args.data_dir, args.data_name, 'dev.tsv'),
        args.max_seq_len
    )
    devloader = DataLoader(
        dev_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    test_data = processor(
        tokenizer,
        os.path.join(args.data_dir, args.data_name, 'test.tsv'),
        args.max_seq_len
    )
    testloader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.task_name == 'sts':
        model = BertForSTS(args)
        loss_fn = torch.nn.MSELoss()
    else:
        raise AssertionError('The assigned task name must be STS.')

    model = model.to(device)
    model.train()

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-06)
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-06, weight_decay=args.weight_decay)

    # learning rate scheduler (linear)
    num_training_steps = int(len(trainloader) * args.epochs)
    num_warmup_steps = int(num_training_steps * args.warmup_proportion)

    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        lr_end=0.0,
        power=1.0,
        last_epoch=-1
    )  #cycle=False

    # scheduler = get_linear_schedule_with_warmup(optimizer,
    #                                             num_warmup_steps=num_warmup_steps,
    #                                             num_training_steps=num_training_steps,
    #                                             last_epoch=-1)  #cycle=False

    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp_name))

    for epoch in range(1, args.epochs+1):
        # start training in each epoch
        train_loss = 0
        dev_history = 0
        for train_batch in trainloader:
            optimizer.zero_grad()

            text = train_batch[0].to(device)
            segments = train_batch[1].to(device)
            attention_masks = train_batch[2].to(device)
            scores = train_batch[3].to(device)

            outputs = model(input_ids=text,
                            attention_mask=attention_masks,
                            token_type_ids=segments)
            loss = loss_fn(outputs.squeeze(-1), scores)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clipping)
            optimizer.step()

            # Update learning rate schedule
            scheduler.step()

            train_loss += loss.item()
            writer.add_scalar(f'train_loss',
                              train_loss,
                              epoch)

        print("Epoch {}, train_loss: {}".format(epoch, train_loss/len(trainloader)))

        if epoch  % 5 == 0:
            # start evaluating
            model.eval()
            print("=========================================")
            dev_loss = 0
            #predictions = torch.empty_like(devloader)
            for dev_batch in devloader:
                text = dev_batch[0].to(device)
                segments = dev_batch[1].to(device)
                attention_masks = dev_batch[2].to(device)
                scores = dev_batch[3].to(device)

                with torch.no_grad():
                    outputs = model(input_ids=text,
                                    attention_mask=attention_masks,
                                    token_type_ids=segments)
                    loss = loss_fn(outputs.squeeze(-1), scores)
                    dev_loss += loss.item()
                    if dev_loss > dev_history and epoch > args.epochs // 2:
                        print("Find a better model! Let's save it.")
                        dev_history = dev_loss
                        torch.save(model.state_dict(), args.save_dir+'/model.pkl')
                    pearson = pearsonr(outputs.squeeze(-1).cpu().numpy(), scores.cpu().numpy())[0]

            print("Epoch {}, valid_loss: {}, pearson:{} ".format(epoch, dev_loss/len(devloader), pearson))

        if epoch  == args.epochs:
            # start testing
            model.load_state_dict(torch.load(args.save_dir+'/model.pkl'))
            model.eval()
            print("=========================================")
            test_loss = 0
            #predictions = torch.empty_like(devloader)
            for test_batch in testloader:
                text = test_batch[0].to(device)
                segments = test_batch[1].to(device)
                attention_masks = test_batch[2].to(device)
                scores = test_batch[3].to(device)

                with torch.no_grad():
                    outputs = model(input_ids=text,
                                    attention_mask=attention_masks,
                                    token_type_ids=segments)
                    loss = loss_fn(outputs.squeeze(-1), scores)
                    test_loss += loss.item()
                    pearson_v = pearsonr(outputs.squeeze(-1).cpu().numpy(), scores.cpu().numpy())[0]

            print("Finished! test_loss: {}, pearson:{} ".format(test_loss/len(testloader), pearson_v))

            

    
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp_name',
        default='lr_5e-5',
        type=str
    )
    parser.add_argument(
        '--model_name',
        default='bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12',
        #default='bert-base-uncased',
        type=str
    )
    parser.add_argument(
        '--task_name',
        default='sts',
        type=str
    )
    parser.add_argument(
        '--data_name',
        default='BIOSSES',
        choices=['BIOSSES', 'clinicalSTS', 'mednli', 'STS-B'],
        type=str
    )
    parser.add_argument(
        '--data_dir',
        default='/home/dean/datasets/benchmarks/BLUE/data_v0.2/data/',
        choices=[
            '/home/dean/datasets/benchmarks/BLUE/data_v0.2/data/',
            '/home/dean/datasets/benchmarks/GLUE/STS-B', # Use STS-B dataset for checking model performance.
            ],
        type=str
    )
    parser.add_argument(
        '--save_dir',
        default='./saved_model',
        type=str
    )
    parser.add_argument(
        '--log_dir',
        default='./logs/sts',
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
        default=30,
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
    parser.add_argument(
        '--grad_clipping',
        default=1.8,
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
