import argparse
import copy
import datetime
import heapq
import os
from pprint import pprint
import sys
import random
import shutil

import numpy as np
import sentencepiece as spm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer

from .parameters import *
from .utils import *
from .transformer import *


def train(_step, stop_signal, train_loader, model, criterion, optim, device, args):
    """Train for one epoch"""
    train_losses = []
    start_time = datetime.datetime.now()

    model.train()
    for i, batch in enumerate(train_loader):
        src_input, trg_input, trg_output = batch
        src_input, trg_input, trg_output = src_input.to(device), trg_input.to(device), trg_output.to(device)

        e_mask, d_mask = make_mask(src_input, trg_input, args)

        output = model(src_input, trg_input, e_mask, d_mask)[0] # (B, L, vocab_size)

        trg_output_shape = trg_output.shape
        optim.zero_grad()
        loss = criterion(
            output.view(-1, args.trg_vocab_size),
            trg_output.view(trg_output_shape[0] * trg_output_shape[1])
        )

        #loss.sum().backward()
        loss.backward()
        optim.step(_step)
        _step += 1

        train_losses.append(loss.item())

        del src_input, trg_input, trg_output, e_mask, d_mask, output
        torch.cuda.empty_cache()

        if _step == len(train_loader):
            args.loss = np.mean(train_losses)
            save_checkpoint(_step, model, optim, args)
        elif _step == train_step:
            save_checkpoint(_step, model, optim, args)
            stop_signal = True
            break
        #elif _step in [ 25000, 50000,75000]:
        elif _step in [ i for i in range(25000, train_step, 25000)]:
            save_checkpoint(_step, model, optim, args)

        #if args.rank == 0:
        #    print(args.rank, 'Lr', optim._get_lr())
    end_time = datetime.datetime.now()
    training_time = end_time - start_time
    seconds = training_time.seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    mean_train_loss = np.mean(train_losses)
    #print(f"#################### Epoch: {epoch} ####################")
    #print(f"Train loss: {mean_train_loss} || One epoch training time: {hours}hrs {minutes}mins {seconds}secs")
    return _step, stop_signal, mean_train_loss,  f"{hours}:{minutes}:{seconds}"


def validation(valid_loader, model, criterion,device, args ):
    valid_losses = []
    start_time = datetime.datetime.now()

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            src_input, trg_input, trg_output = batch
            src_input, trg_input, trg_output = src_input.to(device), trg_input.to(device), trg_output.to(device)

            e_mask, d_mask = make_mask(src_input, trg_input, args)

            output = model(src_input, trg_input, e_mask, d_mask)[0] # (B, L, vocab_size)
            trg_output_shape = trg_output.shape
            loss = criterion(
                output.view(-1, args.trg_vocab_size),
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

    return mean_valid_loss, f"{hours}:{minutes}:{seconds}"


def cleanup():
    dist.destroy_process_group()


def save_checkpoint(_step, model, optimizer, args):
    #print('consolidating...')
    optimizer.consolidate_state_dict(),
    dist.barrier()
    if args.rank == 0:
        print('saving...', _step)
        state_dict = {
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'loss': args.loss
        }
        torch.save(state_dict, f"{ckpt_dir}/{args.fp}_{args.model_type}_checkpoint_ddp_{_step}.pth")
        print('saved!')


def init_processes(rank, world_size, free_port, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = free_port
    print(free_port)

    dist.init_process_group(backend='gloo', init_method='env://', world_size=world_size, rank=rank)


def main_train(rank, args):

    print('rank', rank)
    args.rank = rank

    assert args.ddp
    init_processes(rank, args.world_size, args.port, backend='nccl')
    print(f"{torch.distributed.is_initialized() = } {args.rank}/{args.world_size}")

    #print("Loading Transformer model & Adam optimizer... ")
    model = build_model(args)

    # Define loss function
    best_loss = sys.float_info.max
    args.loss = best_loss
    criterion = nn.NLLLoss().cuda(args.rank)

    if args.ddp and args.sync_bn:
        process_group = torch.distributed.group.WORLD
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)

    torch.cuda.set_device(args.rank)
    model.cuda(args.rank)

    if args.ddp:
        ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.rank])
        optim = ZeroRedundancyOptimizer(ddp_model.parameters(), optimizer_class=torch.optim.Adam, lr=learning_rate, betas=(0.90, 0.98))
        optimizer = CustomOptim(5, 5000, optim)
        #optimizer = ZeroRedundancyOptimizer(ddp_model.parameters(), optim=torch.optim.Adam, lr=learning_rate*args.world_size, betas=(0.90, 0.98))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.90, 0.98))
        #optimizer = torch.optim.Adam(ddp_model.parameters(), lr=learning_rate*args.world_size, betas=(0.90, 0.98))

    if args.resume:
        assert os.path.exists(f"{ckpt_dir}/{args.resume}"), f"There is no checkpoint named {args.resume}."
        print("Loading checkpoint...")
        if rank is None:
            checkpoint = torch.load(f"{ckpt_dir}/{args.resume}")
        else:
            # Map model to be loaded to specified single gpu.
            # configure map_location properly
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            checkpoint = torch.load(f"{ckpt_dir}/{args.resume}", map_location=map_location)

        ddp_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        best_loss = checkpoint['loss']
    else:
        print("Initializing the model...")
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # Data loading code
    train_loader = get_data_loader(f'{data_dir}/fingerprints/{args.fp}.{args.model_type}.{TRAIN_DATA}', args)
    valid_loader = get_data_loader(f'{data_dir}/fingerprints/{args.fp}.{args.model_type}.{VALID_DATA}', args)

    dist.barrier()

    nepochs = round(train_step/len(train_loader))
    print(f"# of parameters of the model: {train_step=} {nepochs=} X {len(train_loader)=}")
    if args.start_step:
        _step = int(args.start_step)
    else:
        _step  = 0

    print('Training started...', _step)
    train_stop = False
    while not train_stop:
        _step, train_stop, train_loss, train_time = train(_step, train_stop, train_loader, ddp_model, criterion, optimizer, args.rank, args)

        if args.rank == 0:
            valid_loss, valid_time = validation(valid_loader, ddp_model.module, criterion, args.rank, args)
            args.loss = valid_loss
            print('---'*15)
            print(f"Step:{_step}/{train_step} \t\tEpoch:{_step/len(train_loader)}/{nepochs}")
            print(f"TrainLoss: {train_loss:.8} \tTrainTime: {train_time} ")
            print(f"ValidLoss: {valid_loss:.8} \tValidTime: {valid_time}")

        dist.barrier()

    if args.ddp:
        cleanup()
    print('Done', args.rank)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp', required=True, help='Name of the fingerprint')
    parser.add_argument('--model_type', required=True, help='The representation of molecules, either "smiles" or "selfies"')

    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--start_step', default=0, type=int, metavar='N',
                        help='manual train step number (useful on restarts)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--sync_bn', action='store_true', help='bn will be synchronized')
    parser.add_argument('--ddp', action='store_true', default=True,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')


    args = parser.parse_args()

    # checking the settings...
    assert args.fp in fp_names, f"Choose one of the fingerprints: \n{fp_names}"
    assert args.model_type in ['smiles', 'selfies'], f"Either 'smiles' or 'selfies'"

    if args.resume:
        assert os.path.exists(f"{ckpt_dir}/{args.resume}"), f"There is no checkpoint with the name {args.resume}."
        #assert os.path.exists(f"{ckpt_dir}/{args.checkpoint_name}"), f"There is no checkpoint named {checkpoint_name}."

    if args.ddp:
        assert torch.distributed.is_available() and  torch.distributed.is_nccl_available(), \
                "PyTorch Distributed module is not available in current system!"

    args.src_vocab_size = fp_vocab_sizes[args.fp]
    args.trg_vocab_size = trg_vocab_sizes[args.model_type]
    args.src_seq_len = fp_seq_lens[args.fp]
    args.trg_seq_len = trg_seq_len
    args.batch_size = 10 #batch_sizes[args.fp]
    args.port = find_free_port()

    args.root_dir = root_dir
    args.fp_datadir = data_dir.joinpath('fingerprints', args.fp)

    args.src_sp_prefix = f"{sp_dir}/{args.fp}_{SP_NAME}"
    args.trg_sp_prefix = f"{sp_dir}/{args.model_type}_{SP_NAME}"

    print('Here we go..')
    [ print(f'{i :>15} : {j}') for i,j in vars(args).items()]
    print(f"")

    if args.ddp:
        assert torch.distributed.is_available() and  torch.distributed.is_nccl_available()
        #world_size is the total number of processes to run;
        args.world_size = torch.cuda.device_count() #world_size = ngpus_per_node * nnodes

        # Use torch.multiprocessing.spawn to launch distributed processes: the main_worker process func.
        print(f'Starting DDP training, {torch.cuda.device_count()} GPUs will be used for training.')
        mp.spawn(main_train, nprocs=args.world_size, args=(args,), join=True)

    else:
        main_train(0, args)
