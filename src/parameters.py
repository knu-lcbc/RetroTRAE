import torch

# Path or parameters for data
DATA_DIR = 'data'
SP_DIR = f'{DATA_DIR}/sp'
SRC_DIR = 'src'
TRG_DIR = 'trg'
SRC_RAW_DATA_NAME = 'raw_data.src'
TRG_RAW_DATA_NAME = 'raw_data.trg'
TRAIN_NAME = 'train.txt'
VALID_NAME = 'valid.txt'
TEST_NAME = 'test.txt'

# Parameters for sentencepiece tokenizer
pad_id = 0 ; sos_id = 1 ; eos_id = 2 ; unk_id = 3
src_model_prefix = 'src_sp'
trg_model_prefix = 'trg_sp'
src_vocab_size = 7492
trg_vocab_size = 7920
character_coverage = 1.0
model_type = 'word'

# Parameters for Transformer & training
learning_rate = 0.001
batch_size = 100
seq_len = 100
num_heads = 8
num_layers = 4
d_model = 1024
d_ff = 2048
d_k = d_model // num_heads
drop_out_rate = 0.1
num_epochs = 1000
beam_size = 10
ckpt_dir = 'saved_model'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
