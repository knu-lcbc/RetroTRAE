from parameters import *
from utils import *
from transformer import *
from predict import *

from torch import nn
import torch

import sys, os
import argparse
import datetime
import copy
import heapq

import sentencepiece as spm
import numpy as np

def setup(resume_training=False, checkpoint_name=None):

    # Load Transformer model & Adam optimizer..
    print("Loading Transformer model & Adam optimizer...")
    model = build_model()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.90, 0.98))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=18000, T_mult=1, eta_min=0.0, last_epoch=-1)
    #print(f"\nInintial scheduler:{scheduler.state_dict()} \n")
    #---

    if resume_training:
        assert os.path.exists(f"{ckpt_dir}/{checkpoint_name}"), f"There is no checkpoint named {checkpoint_name}."

        print("Loading checkpoint...")
        checkpoint = torch.load(f"{ckpt_dir}/{checkpoint_name}")
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optim_state_dict'])
        best_loss = checkpoint['loss']
    else:
        print("Initializing the model...")
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # Load loss function
    print("Loading loss function...")
    criterion = nn.NLLLoss()

    print("Setting finished.")

    return model, optim, scheduler, criterion

def train(resume=False, custom_validation=False, checkpoint_name=None):

    model, optim, scheduler, criterion = setup(resume_training=resume, checkpoint_name=checkpoint_name)

    # Load dataloaders
    print("Loading dataloaders...")
    train_loader = get_data_loader(TRAIN_NAME)
    valid_loader = get_data_loader(VALID_NAME)

    best_loss = sys.float_info.max


    print("Training starts.")
    print("hyperparameters..")
    print(f'\n\tLearning rate:{learning_rate}\n\tBatch_size:{batch_size} \n\tEpochs:{num_epochs} \n\tDropout_rate:{drop_out_rate} \n\tBeamSearch:{beam_size}')
    print(f'\n\tnum_heads:{num_heads}, num_layers:{num_layers}, d_model:{d_model}, d_ff:{d_ff}')


    for epoch in range(1, num_epochs+1):
        model.train()

        train_losses = []
        start_time = datetime.datetime.now()

        for i, batch in enumerate(train_loader):
            src_input, trg_input, trg_output = batch
            src_input, trg_input, trg_output = src_input.to(device), trg_input.to(device), trg_output.to(device)

            e_mask, d_mask = make_mask(src_input, trg_input)

            output = model(src_input, trg_input, e_mask, d_mask) # (B, L, vocab_size)

            trg_output_shape = trg_output.shape
            optim.zero_grad()
            loss = criterion(
                output.view(-1, trg_vocab_size),
                trg_output.view(trg_output_shape[0] * trg_output_shape[1])
            )

            loss.backward()
            optim.step()
            scheduler.step()

            train_losses.append(loss.item())

            del src_input, trg_input, trg_output, e_mask, d_mask, output
            torch.cuda.empty_cache()

        end_time = datetime.datetime.now()
        training_time = end_time - start_time
        seconds = training_time.seconds
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60

        mean_train_loss = np.mean(train_losses)
        print(f"#################### Epoch: {epoch} ####################")
        print(f"Train loss: {mean_train_loss} || One epoch training time: {hours}hrs {minutes}mins {seconds}secs")


        if epoch % 100 == 0 :
            valid_loss, valid_time = validation(model, criterion, valid_loader)
            if custom_validaton:
                print('Custom validation is running...')
                predict(model, valid_loader, method='greedy')

            if not os.path.exists(ckpt_dir):
                os.mkdir(ckpt_dir)
                state_dict = {
                    'model_state_dict': model.state_dict(),
                    'optim_state_dict': optim.state_dict(),
                    'loss': valid_loss
                }
                torch.save(state_dict, f"{ckpt_dir}/checkpoint_epoch_{epoch}.pth")
                print(f"***** Current checkpoint is saved. *****")

            if valid_loss < best_loss:
                state_dict = {
                    'model_state_dict': model.state_dict(),
                    'optim_state_dict': optim.state_dict(),
                    'loss': best_loss
                }
                torch.save(state_dict, f"{ckpt_dir}/best_checkpoint.pth")
                print(f"***** Current best checkpoint is saved. *****")
                best_loss = valid_loss

            print(f"Best valid loss: {best_loss}")
            print(f"Valid loss: {valid_loss} || One epoch training time: {valid_time}")

    print(f"Training finished!")

def validation(model, criterion, valid_loader):

    print("Validation processing...")
    model.eval()

    valid_losses = []
    start_time = datetime.datetime.now()

    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            src_input, trg_input, trg_output = batch
            src_input, trg_input, trg_output = src_input.to(device), trg_input.to(device), trg_output.to(device)

            e_mask, d_mask = make_mask(src_input, trg_input)

            output = model(src_input, trg_input, e_mask, d_mask) # (B, L, vocab_size)
            trg_output_shape = trg_output.shape
            loss = criterion(
                output.view(-1, trg_vocab_size),
                trg_output.view(trg_output_shape[0] * trg_output_shape[1])
            )
            valid_losses.append(loss.item())

            del src_input, trg_input, trg_output, e_mask, d_mask, output
            torch.cuda.empty_cache()

    end_time = datetime.datetime.now()
    validation_time = end_time - start_time
    seconds = validation_time.seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    mean_valid_loss = np.mean(valid_losses)

    return mean_valid_loss, f"{hours}hrs {minutes}mins {seconds}secs"

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', default=False, type=bool, help="Resume: True or Falsee?")
    parser.add_argument('--custom_validation', default=False, type=str, help="Custom validations: True or False")
    parser.add_argument('--checkpoint_name', default=None, type=str, help="checkpoint file")

    args = parser.parse_args()

    train(resume=args.resume, custom_validation=args.custom_validation, checkpoint_name=args.checkpoint_name)
