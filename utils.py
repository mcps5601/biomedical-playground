import torch
import os, random
import numpy as np

# This function can be successfully used in PyTorch v.1.6.0 for reproducibility.
def set_seed(seed):
    """
    Args:
        seed: an integer number to initialize a pseudorandom number generator
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        #torch.cuda.manual_seed_all(seed)  # if using more than one GPUs
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def _truncate_seq_pair(tokens_a, tokens_b, max_seq_len):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_seq_len:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()