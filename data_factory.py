import torch
import pandas as pd
from utils import _truncate_seq_pair, convert_to_unicode


class BiossesDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data_path, max_seq_len):
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.max_seq_len = max_seq_len
        self.df = pd.read_csv(self.data_path, sep="\t")
    
    def __getitem__(self, idx):
        sent1, sent2, score = self.df.iloc[idx, 7:10]
        score_tensor = torch.tensor(score, dtype=torch.float32)

        tokens_1 = self.tokenizer.tokenize(convert_to_unicode(sent1))
        tokens_2 = self.tokenizer.tokenize(convert_to_unicode(sent2))
        _truncate_seq_pair(tokens_1, tokens_2, self.max_seq_len - 3)

        # assemble [CLS]+sent1+[SEP]
        word_pieces = ["[CLS]"]
        word_pieces += tokens_1 + ["[SEP]"]
        len_1 = len(tokens_1) + 2

        # assemble [CLS]+sent1+[SEP]+sent2+[SEP]
        word_pieces += tokens_2 + ["[SEP]"]
        len_2 = len(tokens_2) + 1

        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.LongTensor(ids)
        segments_tensor = torch.LongTensor([0] * len_1 + [1] * len_2)

        return (tokens_tensor, segments_tensor, score_tensor)

    def __len__(self):
        return len(self.df)


class MednliDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data_path, max_seq_len):
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.max_seq_len = max_seq_len
        self.df = pd.read_csv(self.data_path, sep="\t")
        self.label_map = {'contradiction': 0, 'entailment': 1, 'neutral': 2}

    def __getitem__(self, idx):
        label, guid, text_a, text_b = self.df.iloc[idx, :]
        tokens_1 = self.tokenizer.tokenize(convert_to_unicode(text_a))
        tokens_2 = self.tokenizer.tokenize(convert_to_unicode(text_b))
        _truncate_seq_pair(tokens_1, tokens_2, self.max_seq_len - 3)

        # assemble [CLS]+text_a+[SEP]
        word_pieces = ["[CLS]"]
        word_pieces += tokens_1 + ["[SEP]"]
        len_1 = len(tokens_1) + 2

        # assemble [CLS]+sent1+[SEP]+text_b+[SEP]
        word_pieces += tokens_2 + ["[SEP]"]
        len_2 = len(tokens_2) + 1

        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.LongTensor(ids)
        segments_tensor = torch.LongTensor([0] * len_1 + [1] * len_2)

        # labels
        label_id = self.label_map[label]
        label_tensor = torch.tensor(label_id)
        # label_tensor = torch.IntTensor(label_id)

        return (tokens_tensor, segments_tensor, label_tensor)

    def __len__(self):
        return len(self.df)