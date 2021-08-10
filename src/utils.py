from torch.utils.data import Dataset, DataLoader
from parameters import *
from transformer import *

import torch
import sentencepiece as spm
import numpy as np
import heapq

def build_model():
    print("Loading vocabs...")
    src_i2w = {}
    trg_i2w = {}

    with open(f"{SP_DIR}/{src_model_prefix}.vocab", encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        word = line.strip().split('\t')[0]
        src_i2w[i] = word

    with open(f"{SP_DIR}/{trg_model_prefix}.vocab", encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        word = line.strip().split('\t')[0]
        trg_i2w[i] = word

    print(f"The size of src vocab is {len(src_i2w)} and that of trg vocab is {len(trg_i2w)}.")

    model = Transformer(src_vocab_size=len(src_i2w), trg_vocab_size=len(trg_i2w)).to(device)

    return model

def make_mask(src_input, trg_input):
    e_mask = (src_input != pad_id).unsqueeze(1)  # (B, 1, L)
    d_mask = (trg_input != pad_id).unsqueeze(1)  # (B, 1, L)

    nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool)  # (1, L, L)
    nopeak_mask = torch.tril(nopeak_mask).to(device)  # (1, L, L) to triangular shape
    d_mask = d_mask & nopeak_mask  # (B, L, L) padding false

    return e_mask, d_mask

# custom data structure for beam search method
class BeamNode():
    def __init__(self, cur_idx, prob, decoded):
        self.cur_idx = cur_idx
        self.prob = prob
        self.decoded = decoded
        self.is_finished = False

    def __gt__(self, other):
        return self.prob > other.prob

    def __ge__(self, other):
        return self.prob >= other.prob

    def __lt__(self, other):
        return self.prob < other.prob

    def __le__(self, other):
        return self.prob <= other.prob

    def __eq__(self, other):
        return self.prob == other.prob

    def __ne__(self, other):
        return self.prob != other.prob

    def print_spec(self):
        print(f"ID: {self} || cur_idx: {self.cur_idx} || prob: {self.prob} || decoded: {self.decoded}")

class PriorityQueue():

    def __init__(self):
        self.queue = []

    def put(self, obj):
        heapq.heappush(self.queue, (obj.prob, obj))

    def get(self):
        return heapq.heappop(self.queue)[1]

    def qsize(self):
        return len(self.queue)

    def print_scores(self):
        scores = [t[0] for t in self.queue]
        print(scores)

    def print_objs(self):
        objs = [t[1] for t in self.queue]
        print(objs)

#################
# Data loaders
src_sp = spm.SentencePieceProcessor()
trg_sp = spm.SentencePieceProcessor()
src_sp.Load(f"{SP_DIR}/{src_model_prefix}.model")
trg_sp.Load(f"{SP_DIR}/{trg_model_prefix}.model")


def get_data_loader(file_name):
    print(f"Getting source/target {file_name}...")
    with open(f"{DATA_DIR}/{SRC_DIR}/{file_name}", 'r', encoding="utf-8") as f:
        src_text_list = f.readlines()

    with open(f"{DATA_DIR}/{TRG_DIR}/{file_name}", 'r', encoding="utf-8") as f:
        trg_text_list = f.readlines()

    print("Tokenizing & Padding src data...")
    src_list = process_src(src_text_list) # (sample_num, L)
    print(f"The shape of src data: {np.shape(src_list)}")

    print("Tokenizing & Padding trg data...")
    input_trg_list, output_trg_list = process_trg(trg_text_list) # (sample_num, L)
    print(f"The shape of input trg data: {np.shape(input_trg_list)}")
    print(f"The shape of output trg data: {np.shape(output_trg_list)}")

    dataset = CustomDataset(src_list, input_trg_list, output_trg_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

def pad_or_truncate(tokenized_text):
    if len(tokenized_text) < seq_len:
        left = seq_len - len(tokenized_text)
        padding = [pad_id] * left
        tokenized_text += padding
    else:
        tokenized_text = tokenized_text[:seq_len]

    return tokenized_text

def process_src(text_list):
    tokenized_list = []
    for text in text_list:
        tokenized = src_sp.EncodeAsIds(text.strip())
        tokenized_list.append(pad_or_truncate(tokenized + [eos_id]))

    return tokenized_list

def process_trg(text_list):
    input_tokenized_list = []
    output_tokenized_list = []
    for text in text_list:
        tokenized = trg_sp.EncodeAsIds(text.strip())
        trg_input = [sos_id] + tokenized
        trg_output = tokenized + [eos_id]
        input_tokenized_list.append(pad_or_truncate(trg_input))
        output_tokenized_list.append(pad_or_truncate(trg_output))

    return input_tokenized_list, output_tokenized_list

class CustomDataset(Dataset):
    def __init__(self, src_list, input_trg_list, output_trg_list):
        super().__init__()
        self.src_data = torch.LongTensor(src_list)
        self.input_trg_data = torch.LongTensor(input_trg_list)
        self.output_trg_data = torch.LongTensor(output_trg_list)

        assert np.shape(src_list) == np.shape(input_trg_list), "The shape of src_list and input_trg_list are different."
        assert np.shape(input_trg_list) == np.shape(output_trg_list), "The shape of input_trg_list and output_trg_list are different."

    def make_mask(self):
        e_mask = (self.src_data != pad_id).unsqueeze(1) # (num_samples, 1, L)
        d_mask = (self.input_trg_data != pad_id).unsqueeze(1) # (num_samples, 1, L)

        nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool) # (1, L, L)
        nopeak_mask = torch.tril(nopeak_mask) # (1, L, L) to triangular shape
        d_mask = d_mask & nopeak_mask # (num_samples, L, L) padding false

        return e_mask, d_mask

    def __getitem__(self, idx):
        return self.src_data[idx], self.input_trg_data[idx], self.output_trg_data[idx]

    def __len__(self):
        return np.shape(self.src_data)[0]


# Metrics for evaluation
def molGen(input):
    Slist = list()
    input.strip()
    if ' . ' in input:
        input = input.split(' . ')
        R1 = input[0].strip().split()
        R2 = input[1].strip().split()
        Slist.append(R1)
        Slist.append(R2)
    else:
        R = input.split()
        Slist.append(R)
    return Slist

def tanimoto(truth, prediction, i, j):
    return len(set(truth[i]) & set(prediction[j])) / float(len(set(truth[i]) | set(prediction[j])))

def similarity(truth, prediction):
# Sdict = Similarity dictiontionary, Nlist = NameList, Vlist = Value list
    Sdict = dict()
    if len(truth) == 2 and len(prediction) == 2:
        # ground truth A >> B + C. Prediction A >> D + E
        Nlist = ['DB', 'DC', 'EB', 'EC']
        Vlist = [(0,0), (1,0), (0,1), (1,1)]

        for i in range(4):
            Sdict[Nlist[i]] = tanimoto(truth, prediction, Vlist[i][0], Vlist[i][1])

        if Sdict['DB'] >= Sdict['DC']:
            del Sdict['DC']
        else:
            del Sdict['DB']

        if Sdict['EB'] >= Sdict['EC']:
            del Sdict['EC']
        else:
            del Sdict['EB']

    # Condition 2

    elif len(truth) == 1 and len(prediction) == 2:
        # ground truth A >> G. Prediction A >> D + E
        Nlist = ['DG', 'EG']
        Vlist = [(0,0), (0,1)]

        for i in range(2):
            Sdict[Nlist[i]] = tanimoto(truth, prediction, Vlist[i][0], Vlist[i][1])

        if Sdict['DG'] >= Sdict['EG']:
            del Sdict['EG']
        else:
            del Sdict['DG']

    # Condition 3

    elif len(truth) == 2 and len(prediction) == 1:
        # ground truth A >> B + C. Prediction A >> F
        Nlist = ['FB', 'FC']
        Vlist = [(0,0), (1,0)]

        for i in range(2):
            Sdict[Nlist[i]] = tanimoto(truth, prediction, Vlist[i][0], Vlist[i][1])

        if Sdict['FB'] >= Sdict['FC']:
            del Sdict['FC']
        else:
            del Sdict['FB']

    # Condition 4

    elif len(truth) == 1 and len(prediction) == 1:
        # ground truth A >> G. Prediction A >> F
        Nlist = ['FG']
        Vlist = [(0,0)]
        Sdict[Nlist[0]] = tanimoto(truth, prediction, Vlist[0][0], Vlist[0][1])

    else:
        Sdict['Prediction'] = 0

    return Sdict


