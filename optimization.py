from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu):

