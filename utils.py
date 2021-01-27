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