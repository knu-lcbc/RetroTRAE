import argparse
import logging
import sys, os
from pathlib import Path

from RetroTRAE.database import mp_dbGenerate

logging.basicConfig(level=logging.INFO,
                format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                stream=sys.stdout,
               )

logger = logging.getLogger(__file__)

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description ='PubChem database generator for molecule retrieval', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cid-smiles-path', type=str,required=True, help="Raw 'CID-SMILES' file downloaded from PubChem database")
    parser.add_argument('--output-dir', default='./data/PubChem_AEs', type=str, help="Name of a directory to save database files")
    parser.add_argument('--lines-per-file', type=int, default=500000, help="In order to search effectively database file needs to be split to smaller files")
    parser.add_argument('--num-sample', type=int, default=None, help="Number of lines of 'CID-SMILES' file  for quick experimenting")
    parser.add_argument('--rdLogger', action='store_true', default=False, help="Enable RDKit Logger. By default it ignores error messages from the RDKit")
    parser.add_argument('--tmp-dir', default='tmp.data/', type=str, help="An empty directory for temporarily use")
    parser.add_argument('--tmp-file-suffix', default='raw', type=str, help="Suffix for the tmp files in order to track them")
    parser.add_argument('--split_file', action='store_true', default=False, help="Use when running the code for the first time")
    args = parser.parse_args()

    logger.info(args)
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
        
    if not os.path.isdir(args.tmp_dir):
        os.mkdir(args.tmp_dir)
    #aes = getAtomEnvs('CC(=O)OC(CC(=O)[O-])C[N+](C)(C)C', radii=[0, 1], radius=1, nbits=1024, rdLogger=args.rdLogger)

    if args.split_file:
        #First, we split 'CID-SMILES' into small files. It will be easy to handle if the execution is interrupted.
        logger.info(f"First, we split {args.cid_smiles_path} into small files. It will be easy to handle if the execution is interrupted.")
        split_file(args.cid_smiles_path, args.tmp_dir, args.lines_per_file, suffix=args.tmp_file_suffix)
        logger.info(f'{args.cid_smiles_path} has been split into small files and saved in {args.tmp_dir}.')

    logger.info('Start...')

    if not os.path.exists('job_done'):
        with open('job_done','w') as f:
            pass

    files_done = open('job_done').read().split('\n')
    for file in Path(args.tmp_dir).iterdir():
        if file.is_file() and file.name.endswith(args.tmp_file_suffix):
            if file.name in files_done:
                continue
            target_file = os.path.join(args.output_dir, file.name.replace(args.tmp_file_suffix, 'smarts'))
            mp_dbGenerate(args, file, target_file)
            if args.num_smaple == None:
                with open('job_done', 'a') as f:
                    f.write(f'{file.name}\n')
