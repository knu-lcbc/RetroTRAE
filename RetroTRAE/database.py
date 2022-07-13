import argparse
from pathlib import Path
import logging
import os, sys
import time
import random

from rdkit import Chem
from rdkit import DataStructs
from rdkit import RDLogger

from .utils import getAtomEnvs

logger = logging.getLogger(__name__)

def tc(query, nbit):
    a = len(query)
    b = len(nbit)
    c = len(set(query).intersection(nbit))
    if c != 0:
        return c / (a + b - c)
    else:
        return 0

def timeit(f):
    import time
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        seconds = time2 - time1
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        try:
            logger.info(f'{f.__name__} -> Elapsed time: {hours}hrs {minutes}mins {seconds:.3}secs')
        except:
            print(f'{f.__name__} -> Elapsed time: {hours}hrs {minutes}mins {seconds:.3}secs')
        return ret
    return wrap

def db_search(query, file):
    query_set = set(query.strip().split())
    resultq = []
    for i, line in enumerate(open(file), 1):
        cid, smiles, aes_str = line.strip().split('\t')
        aes_set = set(aes_str.strip().split())
        tanimoto = tc(query_set, aes_set)
        if tanimoto >= 0.6:
            resultq.append( [tanimoto, aes_str, smiles, cid])
    return resultq


@timeit
def mp_dbSearch(query, db_dir, topk=3):
    import multiprocessing as mp
    manager = mp.Manager()
    pool = mp.Pool(mp.cpu_count()+2)

    jobs = []
    for file in Path(db_dir).iterdir():
        if file.name.endswith('smarts'):
            job = pool.apply_async(db_search, (query, file, ))
            jobs.append((job))
    results = []
    for job in jobs:
        for candidates in job.get():
            results.append(candidates)

    pool.close()
    pool.join()
    return sorted(results, key= lambda key: key[0], reverse=True)[:topk]


def file_writer(file_path, q):
    '''Listens for message on the q, write to file. '''
    with open(file_path, 'w') as f:
        logger.info(f"{file_path} is created.")
        while 1:
            message = q.get()
            if message=='Done':
                logger.info(f"{file_path} \t{ message}")
                break
            f.write(str(message))
            f.flush()

def worker(cid, smiles, q, rdLogger):
    try:
        atomEnv = getAtomEnvs(smiles, radii=[0, 1], radius=1, nbits=1024, rdLogger=rdLogger)
        if atomEnv:
            #logger.info(atomEnv)
            q.put(f"{cid}\t{smiles}\t{atomEnv}\n")
    except Exception as e:
        logger.debug(f'{cid, e}')

def split_file(file_path, output_dir, lines_per_file, suffix='smarts'):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        logger.info(f'Created a new directory to save split-files: {output_dir}')
    logger.info(f"Split-files will be saved in : {output_dir}")
    smallfile = None
    for lineno, line in enumerate(open(file_path)):
        if lineno % lines_per_file == 0:
            if smallfile:
                smallfile.close()
            small_filename = '{}/{}.{}'.format(output_dir, lineno + lines_per_file, suffix)
            logger.info(f"{small_filename }")
            smallfile = open(small_filename, "w")
        smallfile.write(line)
    if smallfile:
        smallfile.close()


@timeit
def mp_dbGenerate(args, raw_cid_smiles, file_path):
    import multiprocessing as mp

    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(mp.cpu_count() + 2)

    watcher = pool.apply_async(file_writer, (file_path, q))
    jobs = []
    for i, line in enumerate(open(raw_cid_smiles), 1):
        cid, smiles = line.strip().split('\t')
        job = pool.apply_async(worker, (cid, smiles, q, args.rdLogger))
        jobs.append(job)
        if args.num_sample:
            args.lines_per_file = args.num_sample//5
            if i >= args.num_sample:
                break
            if i % args.lines_per_file ==0:
                logger.info(f'Processing...Line:{ i}')
        else:
            if i % args.lines_per_file ==0:
                logger.info(f'Processing...Line:{ i}')

    for job in jobs:
        job.get()

    q.put('Done')
    pool.close()
    pool.join()

    #split_file(args.tmp_file, args.output_dir, args.lines_per_file)
    logger.info('Done!')

