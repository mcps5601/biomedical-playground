from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import torch
import os, random
import numpy as np
from ignite.engines.engine import Engine

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


def convert_to_unicode(text):
    ### This function is from Google BERT.
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")



def create_supervised_trainer(model, optimizer, loss_fn, device):
    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()
        text = batch[0].to(device)
        segments = batch[1].to(device)
        attention_masks = batch[2].to(device)
        scores = batch[3].to(device)
        outputs = model(input_ids=text,
                        token_type_ids=segments,
                        attention_mask=attention_masks)
        loss = loss_fn(outputs.squeeze(-1), scores)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return Engine(train_step)


def create_supervised_evaluator(model, metrics=None, loss_fn, device):
    def inference_step(engine, batch):
        model.eval()

    return Engine(inference_step)
