import torch
import pandas as pd

class BiossesDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data_path):
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.df = pd.read_csv(self.data_path, sep="\t")
    
    def __getitem__(self, idx):
        sent1, sent2, score = self.df.iloc[idx, 7:10]
        score_tensor = torch.tensor(score, dtype=torch.float32)

        word_pieces = ["[CLS]"]
        tokens_1 = self.tokenizer.tokenize(sent1)
        word_pieces += tokens_1 + ["[SEP]"]
        len_1 = len(tokens_1) + 2

        tokens_2 = self.tokenizer.tokenize(sent2)
        word_pieces += tokens_2 + ["[SEP]"]
        len_2 = len(tokens_2) + 1

        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        segments_tensor = torch.tensor([0] * len_1 + [1] * len_2, 
                                        dtype=torch.long)

        return (tokens_tensor, segments_tensor, score_tensor)

    def __len__(self):
        return len(self.df)
