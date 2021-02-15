import os, json, glob, re
import pandas as pd

folder = '/data/datasets/benchmarks/BLUE/data_v0.2/data/mednli/Original'
data_paths = glob.glob(os.path.join(folder, "*.jsonl"))

choices = ['dev', 'test', 'train']
new_index = ['gold_label', 'pairID', 'sentence1', 'sentence2']

for data, mode in zip(data_paths, choices):
    jsonObj = pd.read_json(path_or_buf=data, lines=True)
    jsonObj = jsonObj.reindex(columns=new_index)
    # print(jsonObj)
    jsonObj.to_csv(os.path.join(folder, mode + ".tsv"), sep="\t", index=False)