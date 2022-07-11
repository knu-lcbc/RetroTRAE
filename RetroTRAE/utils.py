from .parameters import *
from .transformer import *
#from .fingerprints import *

import argparse
from datetime import datetime
import heapq
import os
import logging
import re
import sys
import warnings

import numpy as np
import sentencepiece as spm
#import selfies as sf

from rdkit import Chem
from rdkit import RDLogger
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# --- utils ---
# 1. Models
# 2. Metrics
# 3. DataLoader & Dataset
# 4. Custom LR scheduler/optim
# 5. Miscellaneous


# -------------------
# 0. Logger
logger = logging.getLogger(__name__)

# -------------------
# 1. Models

def build_model(src_sp_prefix, trg_sp_prefix, src_seq_len, trg_seq_len, device, checkpoint_path=None):
    #print("Loading vocabs...")
    src_i2w = {}
    trg_i2w = {}

    with open(f"{src_sp_prefix}.vocab", encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        word = line.strip().split('\t')[0]
        src_i2w[i] = word

    with open(f"{trg_sp_prefix}.vocab", encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        word = line.strip().split('\t')[0]
        trg_i2w[i] = word

    logger.info(f"Building model.\tThe size of src vocab is {len(src_i2w)} and that of trg vocab is {len(trg_i2w)}.")
    #logger.info(f"Building model from {src_sp_prefix, trg_sp_prefix}.\tThe size of src vocab is {len(src_i2w)} and that of trg vocab is {len(trg_i2w)}.")
    model = Transformer(src_vocab_size=len(src_i2w), trg_vocab_size=len(trg_i2w),
                        src_seq_len=src_seq_len,
                        trg_seq_len=trg_seq_len,
                        device=device)

    if checkpoint_path:
        assert os.path.exists(checkpoint_path), f"Can't find the checkpoint: {checkpoint_path}"

        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        updated_checkpoint= {}
        for key, val in checkpoint['model_state_dict'].items():
            if 'module.' in key:
                updated_checkpoint[key.replace('module.', '')] = val
            else:
                updated_checkpoint[key] = val
        model.load_state_dict(updated_checkpoint)

    return model.to(device)


def setup(model, checkpoint_path, device):
    logger.info(f"Loading checkpoint for the model: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    updated_checkpoint= {}
    for key, val in checkpoint['model_state_dict'].items():
        if 'module.' in key:
            updated_checkpoint[key.replace('module.', '')] = val
        else:
            updated_checkpoint[key] = val
    model.load_state_dict(updated_checkpoint)

    return model.to(device)


def make_mask(src_input, trg_input, args):
    e_mask = (src_input != pad_id).unsqueeze(1)  # (B, 1, L)
    d_mask = (trg_input != pad_id).unsqueeze(1)  # (B, 1, L)

    nopeak_mask = torch.ones([1, trg_seq_len, trg_seq_len], dtype=torch.bool)  # (1, L, L)
    nopeak_mask = torch.tril(nopeak_mask).to(args.rank)  # (1, L, L) to triangular shape
    d_mask = d_mask & nopeak_mask  # (B, L, L) padding false

    return e_mask, d_mask


# -----------------
# 2. Metrics

def FpSimilarity(smiles1, smiles2,
                 metric=DataStructs.TanimotoSimilarity, #DiceSimilarity
                 fingerprint=rdMolDescriptors.GetMorganFingerprint,
                 rdLogger=False, # RDKit logger
                 **kwargs,):
    RDLogger.EnableLog('rdApp.*') if rdLogger else RDLogger.DisableLog('rdApp.*')
    try:
        mol1 = Chem.MolFromSmiles(smiles1, sanitize=True)
        mol2 = Chem.MolFromSmiles(smiles2, sanitize=True)
        if mol2 is not None and mol1 is not None:
            fp1 = fingerprint(mol1, **kwargs)
            fp2 = fingerprint(mol2, **kwargs)
            return metric(fp1, fp2)
        else:
            if rdLogger:
                warnings.warn(f'{smiles1=}, {smiles2=}')
            return 0
    except:
        return 0


def fpBitVec(mol, radius=1, nBits=2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol,radius,nBits)


def morganfpTc(truth_smi, pred_smi):
    RDLogger.DisableLog('rdApp.*')
    try:
        pred_mol = Chem.MolFromSmiles(pred_smi)
        if pred_mol is None:
            return 0
        truth_mol = Chem.MolFromSmiles(truth_smi)
        if truth_mol is None:
            return 0
    except:
        return 0
    return DataStructs.TanimotoSimilarity(fpBitVec(truth_mol), fpBitVec(pred_mol))


def Tanimoto_coeff(s1, s2):
    c = len(set(s1)&set(s2))
    return float(c) / (len(s1) + len(s2) - c)


# -----------------
# 3. DataLoaders & Dataset

def get_data_loader(file_path, args):
    src_sp = spm.SentencePieceProcessor()
    trg_sp = spm.SentencePieceProcessor()
    src_sp.Load(f"{args.src_sp_prefix}.model")
    trg_sp.Load(f"{args.trg_sp_prefix}.model")

    src_text_list,  trg_text_list  = [], []
    #print(f"Getting source/target {file_name}...")
    with open(f"{file_path}", 'r', encoding="utf-8") as f:
        for line in f:
            t, s = line.strip().split('\t')
            src_text_list.append(s)
            trg_text_list.append(t)

            if len(src_text_list) == 1000:
                break

    #print("Tokenizing & Padding src data...")
    src_list = process_src(src_text_list, args.src_seq_len, src_sp) # (sample_num, L)
    #print(f"The shape of src data: {np.shape(src_list)}")

    #print("Tokenizing & Padding trg data...")
    input_trg_list, output_trg_list = process_trg(trg_text_list, args.trg_seq_len, trg_sp) # (sample_num, L)
    #print(f"The shape of input trg data: {np.shape(input_trg_list)}")
    #print(f"The shape of output trg data: {np.shape(output_trg_list)}")

    dataset = CustomDataset(src_list, input_trg_list, output_trg_list)
    if args.ddp:

        sampler = DistributedSampler(dataset, rank=args.rank, num_replicas=args.world_size)
        dataloader = DataLoader(dataset, batch_size=args.batch_size//args.world_size, shuffle=False, sampler=sampler, num_workers=args.workers )
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    return dataloader

def pad_or_truncate(tokenized_text, seq_len):
    if len(tokenized_text) < seq_len:
        left = seq_len - len(tokenized_text)
        padding = [pad_id] * left
        tokenized_text += padding
    else:
        tokenized_text = tokenized_text[:seq_len]

    return tokenized_text

def process_src(text_list, seq_len, src_sp):
    tokenized_list = []
    for text in text_list:
        tokenized = src_sp.EncodeAsIds(text.strip())
        tokenized_list.append(pad_or_truncate(tokenized + [eos_id], seq_len))

    return tokenized_list

def process_trg(text_list, seq_len, trg_sp):
    input_tokenized_list = []
    output_tokenized_list = []
    for text in text_list:
        tokenized = trg_sp.EncodeAsIds(text.strip())
        trg_input = [sos_id] + tokenized
        trg_output = tokenized + [eos_id]
        input_tokenized_list.append(pad_or_truncate(trg_input, seq_len))
        output_tokenized_list.append(pad_or_truncate(trg_output, seq_len))

    return input_tokenized_list, output_tokenized_list

class CustomDataset(Dataset):
    def __init__(self, src_list, input_trg_list, output_trg_list):
        super().__init__()
        self.src_data = torch.LongTensor(src_list)
        self.input_trg_data = torch.LongTensor(input_trg_list)
        self.output_trg_data = torch.LongTensor(output_trg_list)

        assert np.shape(src_list)[0] == np.shape(input_trg_list)[0], "The shape of src_list and input_trg_list are different."
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


# -----------------
# 4. Custom LR scheduler/optim

class CustomOptim():
    "Optim wrapper that implement cycling learning rates"
    def __init__(self, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor

    def step(self, i):
        rate = self.rate(i)
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.optimizer.step()

    def rate(self, i):
        self._step += 1
        if (i) % 50000 == 0 and i > 0:
            self._step = self.warmup -1
        return self.factor * min(1.0, self._step / self.warmup) / max(self._step, self.warmup)

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def _get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def consolidate_state_dict(self,):
        self.optimizer.consolidate_state_dict()


# ----------------------
# 5. Preprocessing of input SMILES

def getSmarts(mol,atomID,radius):
    if radius>0:
        env = Chem.FindAtomEnvironmentOfRadiusN(mol,radius,atomID)
        atomsToUse=[]
        for b in env:
            atomsToUse.append(mol.GetBondWithIdx(b).GetBeginAtomIdx())
            atomsToUse.append(mol.GetBondWithIdx(b).GetEndAtomIdx())
        atomsToUse = list(set(atomsToUse))
    else:
        atomsToUse = [atomID]
        env=None
    symbols = []
    for atom in mol.GetAtoms():
        deg = atom.GetDegree()
        isInRing = atom.IsInRing()
        nHs = atom.GetTotalNumHs()
        symbol = '['+atom.GetSmarts()
        if nHs:
            symbol += 'H'
            if nHs>1:
                symbol += '%d'%nHs
        if isInRing:
            symbol += ';R'
        else:
            symbol += ';!R'
        symbol += ';D%d'%deg
        symbol += "]"
        symbols.append(symbol)
    try:
        smart = Chem.MolFragmentToSmiles(mol,atomsToUse,bondsToUse=env,atomSymbols=symbols, allBondsExplicit=True, rootedAtAtom=atomID)
    except (ValueError, RuntimeError) as ve:
        print('atom to use error or precondition bond error')
        return None
    return smart


def getAtomEnvs(smiles, radii=[0, 1], radius=1, nbits=1024, rdLogger=False):
    """
    A function to extract atom environments from the molecular SMILES.

    Parameters
    ----------
    smiles: str
        Molecular SMILES
    radii: list
        list of radii you would like to obtain atom envs.
    radius: int
        radius of MorganFingerprint
    nbits: int
        size of bit vector for MorganFingerprint

    Returns
    -------
    tuple
        a list of atom envs and a string type of this list
    """

    assert max(radii) <= radius, f"the maximum of radii should be equal or lower than radius, but got {max(radius)}"

    RDLogger.EnableLog('rdApp.*') if rdLogger else RDLogger.DisableLog('rdApp.*')
    molP = Chem.MolFromSmiles(smiles.strip())
    if molP is None:
        if rdLogger:
            warnings.warn(f"There is a semantic error in {smiles}")
        #raise Exception (f"There is a semantic error in {smiles}")
        return None

    sanitFail = Chem.SanitizeMol(molP, catchErrors=True)
    if sanitFail:
        if rdLogger:
            warnings.warn(f"Couldn't sanitize: {smiles}")
        #raise Exception (f"Couldn't sanitize: {smiles}")
        return None

    info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(molP,radius=radius, nBits=nbits, bitInfo=info)# condition can change

    info_temp = []
    for bitId,atoms in info.items():
        exampleAtom,exampleRadius = atoms[0]
        description = getSmarts(molP,exampleAtom,exampleRadius)
        if description is None:
            return None
        info_temp.append((bitId, exampleRadius, description))

    #collect the desired output in another list
    updateInfoTemp = []
    for k,j in enumerate(info_temp):
        if j[1] in radii:                           # condition can change
            updateInfoTemp.append(j)
        else:
            continue

    tokens_str = ''
    tokens_list = []
    for k,j in enumerate(updateInfoTemp):
        tokens_str += str(updateInfoTemp[k][2]) + ' ' #[2]-> selecting SMARTS description
        tokens_list.append(str(updateInfoTemp[k][2]))  # condition can change

    #return tokens_list, tokens_str.strip()
    return tokens_str.strip()


# ----------------
# 6. Miscellaneous

def smiles_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    # return ' '.join(tokens), len(tokens)
    return tokens


def selfies_tokenizer(selfies):
    tokens = list(sf.split_selfies(selfies))
    #return ' '.join(tokens), sf.len_selfies(selfies)
    return tokens


def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol


def smiles_with_atom_idx(smiles):
    mol = mol_with_atom_index(Chem.MolFromSmiles(smiles))
    return Chem.MolToSmiles(mol)


def getIdxSmarts(mol):
    info = {}
    fp = GetMorganFingerprint(mol, radius=1, bitInfo=info )

    info_temp = {}
    for bitId,atoms in info.items():
        exampleAtom,exampleRadius = atoms[0]
        description = getSmarts(mol,exampleAtom,exampleRadius)
        info_temp[bitId] = (exampleRadius,  description[-2], description[-1])

    return info_temp


def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def download_checkpoints():
    import gdown
    import tarfile
    from .evaluate import test_evaluate
    from .parameters import root_dir

    # https://stackoverflow.com/questions/56857900/download-shared-google-drive-folder-with-python
    # https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive/39225039#39225039
    # https://drive.google.com/file/d/1qD8JicIwjyxKKLYahKtbBzKhnUq0JBKx/view?usp=sharing
    url =  'https://drive.google.com/uc?id=1qD8JicIwjyxKKLYahKtbBzKhnUq0JBKx'
    file_id = '1qD8JicIwjyxKKLYahKtbBzKhnUq0JBKx'
    output = root_dir.joinpath('saved_models.tar.gz')

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    gdown.download(url, output, quiet=False)
    print('The trained models are successfully downloaded.\n')

    print('*** Extracting archive file ***\n')
    cmd = 'tar -xvzf saved_models.tar.gz'
    os.system(cmd)

    print('--'*5, 'Testing checkpoints', '--'*5)
    print()
    test_evaluate(args)

