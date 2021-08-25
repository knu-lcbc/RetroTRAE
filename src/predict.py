from parameters import *
from utils import *
from transformer import *

from torch import nn
import torch

import sys, os
import argparse
import datetime
import copy
import heapq

import sentencepiece as spm
import numpy as np


def setup(model, checkpoint_name):
    assert os.path.exists(f"{ckpt_dir}/{checkpoint_name}"), f"There is no checkpoint named {checkpoint_name}."

    print("Loading checkpoint...")
    checkpoint = torch.load(f"{ckpt_dir}/{checkpoint_name}")
    model.load_state_dict(checkpoint['model_state_dict'])
    #optim.load_state_dict(checkpoint['optim_state_dict'])
    #best_loss = checkpoint['loss']

    return model

def custom_validation_fn(model, test_loader, method='greedy'):
    start_time = datetime.datetime.now()
    scores = list()

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            src_input, trg_input, trg_output = batch
            src_input, trg_input, trg_output = src_input.to(device), trg_input.to(device), trg_output.to(device)

            #e_mask, d_mask = make_mask(src_input, trg_input)
            for j in  range(len(src_input)):
                # preparing src data for encoder
                src_j = src_input[j].unsqueeze(0).to(device) # (L) => (1, L)
                encoder_mask = (src_j != pad_id).unsqueeze(1).to(device) # (1, L) => (1, 1, L)
                # encoding src input
                src_j = model.src_embedding(src_j) # (1, L) => (1, L, d_model)
                src_j = model.positional_encoder(src_j) # (1, L, d_model)
                encoder_output = model.encoder(src_j, encoder_mask) # (1, L, d_model)
                if method == 'greedy':
                    s_pred = greedy_search(model, encoder_output, encoder_mask, trg_sp)
                elif method == 'beam':
                    s_pred = beam_search(model, encoder_output, encoder_mask, trg_sp)

                s_src   = src_sp.decode_ids(src_input[j].tolist())
                s_truth = trg_sp.decode_ids(trg_output[j].tolist())

                print( "product : ", s_src)
                print( "reactants : ", s_truth)
                print( "predicted : ", s_pred)

                gtruth = molGen(s_truth)
                candidate = molGen(s_pred)
                Sdict = similarity(gtruth, candidate)
                for item in Sdict.items():
                    scores.append(item)
                    print(item)
                #print(scores)
    sum = 0; count = 0; thresh = 0; bad = 0; zeros = 0; seven = 0; five = 0

    for i in range(len(scores)):
        sum += scores[i][1]
        if scores[i][1] == 1:
            count += 1
        elif scores[i][1] >= 0.85:
            thresh += 1
        elif scores[i][1] == 0:
            zeros += 1
        elif scores[i][1] >= 0.70:
            seven += 1
        elif scores[i][1] >= 0.50:
            five += 1
        else:
            bad += 1

    print('---')
    print('exact-goods-sevens-fives-bads-zeros', count, thresh, seven, five, bad, zeros)

    end_time = datetime.datetime.now()
    validation_time = end_time - start_time
    seconds = validation_time.seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    elapsed_time = f"{hours}hrs {minutes}mins {seconds}secs"

    print(f"{elapsed_time}")

def inference(model, input_sentence, method):

    print("Preprocessing input sentence...")
    tokenized = src_sp.EncodeAsIds(input_sentence)
    src_data = torch.LongTensor(pad_or_truncate(tokenized)).unsqueeze(0).to(device) # (1, L)
    e_mask = (src_data != pad_id).unsqueeze(1).to(device) # (1, 1, L)

    start_time = datetime.datetime.now()

    print("Encoding input sentence...")
    src_data = model.src_embedding(src_data)
    src_data = model.positional_encoder(src_data)
    e_output = model.encoder(src_data, e_mask) # (1, L, d_model)

    if method == 'greedy':
        print("Greedy decoding selected.")
        result = greedy_search(model, e_output, e_mask, trg_sp)
    elif method == 'beam':
        print("Beam search selected.")
        result = beam_search(model, e_output, e_mask, trg_sp)

    end_time = datetime.datetime.now()

    total_inference_time = end_time - start_time
    seconds = total_inference_time.seconds
    minutes = seconds // 60
    seconds = seconds % 60

    print(f"Input: {input_sentence}")
    print(f"Result: {result}")
    print(f"Inference finished! || Total inference time: {minutes}mins {seconds}secs")


def greedy_search(model, e_output, e_mask, trg_sp):
    last_words = torch.LongTensor([pad_id] * seq_len).to(device) # (L)
    last_words[0] = sos_id # (L)
    cur_len = 1

    for i in range(seq_len):
        d_mask = (last_words.unsqueeze(0) != pad_id).unsqueeze(1).to(device) # (1, 1, L)
        nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool).to(device)  # (1, L, L)
        nopeak_mask = torch.tril(nopeak_mask)  # (1, L, L) to triangular shape
        d_mask = d_mask & nopeak_mask  # (1, L, L) padding false

        trg_embedded = model.trg_embedding(last_words.unsqueeze(0))
        trg_positional_encoded = model.positional_encoder(trg_embedded)
        decoder_output = model.decoder(
            trg_positional_encoded,
            e_output,
            e_mask,
            d_mask
        ) # (1, L, d_model)

        output = model.softmax(
            model.output_linear(decoder_output)
        ) # (1, L, trg_vocab_size)

        output = torch.argmax(output, dim=-1) # (1, L)
        last_word_id = output[0][i].item()

        if i < seq_len-1:
            last_words[i+1] = last_word_id
            cur_len += 1

        if last_word_id == eos_id:
            break

    if last_words[-1].item() == pad_id:
        decoded_output = last_words[1:cur_len].tolist()
    else:
        decoded_output = last_words[1:].tolist()
    decoded_output = trg_sp.decode_ids(decoded_output)

    return decoded_output

def beam_search(model, e_output, e_mask, trg_sp):
    cur_queue = PriorityQueue()
    for k in range(beam_size):
        cur_queue.put(BeamNode(sos_id, -0.0, [sos_id]))

    finished_count = 0

    for pos in range(seq_len):
        new_queue = PriorityQueue()
        for k in range(beam_size):
            node = cur_queue.get()
            if node.is_finished:
                new_queue.put(node)
            else:
                trg_input = torch.LongTensor(node.decoded + [pad_id] * (seq_len - len(node.decoded))).to(device) # (L)
                d_mask = (trg_input.unsqueeze(0) != pad_id).unsqueeze(1).to(device) # (1, 1, L)
                nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool).to(device)
                nopeak_mask = torch.tril(nopeak_mask) # (1, L, L) to triangular shape
                d_mask = d_mask & nopeak_mask # (1, L, L) padding false

                trg_embedded = model.trg_embedding(trg_input.unsqueeze(0))
                trg_positional_encoded = model.positional_encoder(trg_embedded)
                decoder_output = model.decoder(
                    trg_positional_encoded,
                    e_output,
                    e_mask,
                    d_mask
                ) # (1, L, d_model)

                output = model.softmax(
                    model.output_linear(decoder_output)
                ) # (1, L, trg_vocab_size)

                output = torch.topk(output[0][pos], dim=-1, k=beam_size)
                last_word_ids = output.indices.tolist() # (k)
                last_word_prob = output.values.tolist() # (k)

                for i, idx in enumerate(last_word_ids):
                    new_node = BeamNode(idx, -(-node.prob + last_word_prob[i]), node.decoded + [idx])
                    if idx == eos_id:
                        new_node.prob = new_node.prob / float(len(new_node.decoded))
                        new_node.is_finished = True
                        finished_count += 1
                    new_queue.put(new_node)

        cur_queue = copy.deepcopy(new_queue)

        if finished_count == beam_size:
            break

    decoded_output = cur_queue.get().decoded

    if decoded_output[-1] == eos_id:
        decoded_output = decoded_output[1:-1]
    else:
        decoded_output = decoded_output[1:]

    return trg_sp.decode_ids(decoded_output)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='An input sequence')
    parser.add_argument('--decode', type=str, required=True, default='greedy', help="greedy or beam?")
    parser.add_argument('--checkpoint_name', type=str, required=True, default='best_checkpoint.pth', help="checkpoint file")

    args = parser.parse_args()

    model = setup(build_model(), args.checkpoint_name)

    assert args.decode == 'greedy' or args.decode =='beam', "Please specify correct decoding method, either 'greedy' or 'beam'."

    if args.input:
        inference(model, args.intput_type, args.decode)
    else:
        print("Please enter input sequence.")

