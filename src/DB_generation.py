#!/usr/bin/python3
from utils import getAtomEnvs

import os
import time
import random

from rdkit import Chem
from rdkit import DataStructs
from rdkit import RDLogger

import argparse


def file_writer(file_path, q):
    '''Listens for message on the q, write to file. '''
    with open(file_path, 'w') as f:
        print(file_path, 'is created.')
        while 1:
            message = q.get()
            if message=='kill':
                print(file_path, message)
                break
            f.write(str(message))
            f.flush()

def worker(args, cid, smiles, q):
    atomEnv = getAtomEnvs(smiles, radii=[0, 1], radius=1, nbits=1024, rdLogger=args.rdLogger)
    try:
        if atomEnv:
            q.put(f"{cid}\t{smiles}\t{atomEnv}\n")
    except Exception as e:
        print(cid, e)


def split_file(file_path, dest_path, lines_per_file=500, ):
    print("Chunk files will be saved in :", dest_path)
    smallfile = None
    for lineno, line in enumerate(open(file_path)):
        if lineno % lines_per_file == 0:
            if smallfile:
                smallfile.close()
            small_filename = '{}/{}.smarts'.format(dest_path, lineno + lines_per_file)
            print(file_filename)
            smallfile = open(small_filename, "w")
        smallfile.write(line)
    if smallfile:
        smallfile.close()


def main(args):
    import multiprocessing as mp

    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(mp.cpu_count() + 2)

    watcher = pool.apply_async(file_writer, (f"data/CID-SMILES.smarts", q))
    for i, line in enumerate(open(args.raw_file)):
        cid, smiles = line.strip().split('\t')
        pool.apply_async(worker, (args, cid, smiles, q))
        i = i+1
        if i % 100000==0:
            print('Processing...Line:', i)


    q.put('kill')
    pool.close()
    pool.join()

    #split_file('data/CID-SMILES.smarts', args.dest, 500)
    split_file('data/CID-SMILES.smarts', args.dest, 5000000)
    print('Done!')

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description = 'segment name')
    parser.add_argument('--raw_file', type=str, help="Raw 'CID-SMILES' file downloaded from PubChem database")
    parser.add_argument('--dest', default='pubchem_AEs', type=str, help="Destination directory to save resulting from transformation")
    parser.add_argument('--rdLogger', action='store_true', default=False, help="RDKit Logger. Useful to ignore error messages when disabled.")
    args = parser.parse_args()

    print(args)
    if not os.path.isdir(args.dest):
        os.mkdir(args.dest)
    #aes = getAtomEnvs('CC(=O)OC(CC(=O)[O-])C[N+](C)(C)C', radii=[0, 1], radius=1, nbits=1024, rdLogger=args.rdLogger)

    print('\nStart...')
    main(args)

