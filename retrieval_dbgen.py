#!/usr/bin/python3

import os
import numpy as np
import rdkit
import time
import matplotlib.pyplot as plt
import pandas as pd

import itertools
from collections import Counter
import pickle

from rdkit import Chem
from rdkit.Chem import Draw

from rdkit.Chem.Draw import SimilarityMaps
from IPython.display import SVG
from rdkit.Chem.Draw import rdMolDraw2D
import io
#from PIL import Image

#from rdkit.Chem.Draw import DrawingOptions
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
from rdkit.Chem.AllChem import Compute2DCoords
from rdkit.Chem.Draw import IPythonConsole
from rdkit import rdBase

import argparse

parser = argparse.ArgumentParser(description = 'segment name')
parser.add_argument('fnameR', type = str)
parser.add_argument('fnameW', type = str)
args = parser.parse_args()

#
def getSubstructSmi(mol,atomID,radius):
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
        return
    return smart

# transformation begins..
def transformation(fnameW, fnameR):
    with open(fnameW, 'a') as fnew:
        with open(fnameR, 'r') as file:
            for z, line in enumerate(file):
                #molecule = line.strip()
                molecule = line.strip()
                molP = Chem.MolFromSmiles(molecule)
                if molP is None: continue
                sanitFail = Chem.SanitizeMol(molP, catchErrors=True)
                if sanitFail:
                    continue
                info = {}
                fp = AllChem.GetMorganFingerprintAsBitVect(molP,radius=1,nBits=1024,bitInfo=info)# condition can change
                info_temp = []

                for bitId,atoms in info.items():
                    exampleAtom,exampleRadius = atoms[0]
                    description = getSubstructSmi(molP,exampleAtom,exampleRadius)
                    info_temp.append((bitId, exampleRadius, description))
               
                #collect the desired output in another list
                updateInfoTemp = []
                for k,j in enumerate(info_temp):
                    if j[1] == 1 or j[1] == 0:                           # condition can change
                        updateInfoTemp.append(j)
                    else:
                        continue
                                         
                fnew.write(molecule + '\t')  
                for k,j in enumerate(updateInfoTemp):
                    
                    if j != updateInfoTemp[-1]:
                        fnew.write(str(updateInfoTemp[k][2]) + ' ')  # condition can change
                    else:
                        fnew.write(str(updateInfoTemp[k][2]))  # condition can change

                fnew.write('\n')

transformation(args.fnameW,args.fnameR)


