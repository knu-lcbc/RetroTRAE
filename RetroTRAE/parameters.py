import os
from pathlib import Path
#import torch

# Path or parameters for data
root_dir = Path('./').resolve()
data_dir = root_dir.joinpath('data')
sp_dir = root_dir.joinpath('data', 'sp')

TRAIN_DATA = 'test'
VALID_DATA = 'test'
TEST_DATA = 'test'
SP_NAME = 'vocab_sp'

# Parameters for sentencepiece tokenizer
pad_id = 0
sos_id = 1
eos_id = 2
unk_id = 3
character_coverage = 1.0
sp_model_type = 'word'

# Parameters for Transformer & training
num_heads = 8
num_layers = 6
dim_model = 512
dim_ff = 2048
dim_k = dim_model // num_heads
dropout_rate = 0.1

train_step = 500000
beam_size = 10
learning_rate = 0.001

ckpt_dir = root_dir.joinpath('saved_models')
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)
