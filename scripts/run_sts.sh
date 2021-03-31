#!/bin/bash

# frequently-adjusted variables
dataset="BIOSSES"
learning_rate=5e-5
grad_clipping=1.8
exp="${dataset}_lr${learning_rate}_gc${grad_clipping}"

# main script
python run_sts.py \
--exp_name          $exp \
--model_name        "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12" \
--task_name         "sts" \
--data_name         $dataset \
--data_dir          "/home/dean/datasets/benchmarks/BLUE/data_v0.2/data/" \
--save_dir          "./saved_model" \
--log_dir           "./logs/sts" \
--seed              777 \
--batch_size        32 \
--epochs            30 \
--criteria          "mse" \
--learning_rate     $learning_rate \
--max_seq_len       128 \
--warmup_proportion 0.1 \
--weight_decay      0.01 \
--grad_clipping     $grad_clipping \
