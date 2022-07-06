import argparse
import logging
import sys, os

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
    parser.add_argument('--lines-per-file', type=int, default=500000, help="In order to search effectively PubChem database needs to be split to smaller files")
    parser.add_argument('--num-sample', type=int, default=None, help="Number of lines of 'CID-SMILES' file  for quick experimenting")
    parser.add_argument('--rdLogger', action='store_true', default=False, help="Enable RDKit Logger. By default it ignores error messages from the RDKit")
    parser.add_argument('--tmp-file', default='/tmp/PubChem_AEs', type=str, help="A file name for temporarily use")
    args = parser.parse_args()

    logger.info(args)
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    #aes = getAtomEnvs('CC(=O)OC(CC(=O)[O-])C[N+](C)(C)C', radii=[0, 1], radius=1, nbits=1024, rdLogger=args.rdLogger)

    logger.info('Start...')
    mp_dbGenerate(args)
