import argparse
from datetime import datetime
import logging
import sys, os

from RetroTRAE import configs
from RetroTRAE import inference
from RetroTRAE import utils
from RetroTRAE import mp_dbSearch

import torch

logging.basicConfig(level=logging.INFO,
                format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                stream=sys.stdout,
               )

logger = logging.getLogger(__file__)


def predict(input, retro_model, aes2smiles_model, args, **kwargs):
    # RetroTRAE predictions
    predicted_aes = inference(retro_model, input, method=args.decode, beam_size=args.beam_size, device=args.device, **kwargs)
    logger.info(f"{input=}")
    logger.info(f"RetroTRAE output: {predicted_aes}")

    # convert AEs to SMILES
    smiles_dict = {}
    for aes in predicted_aes.split(' . '):
        if args.conversion =='ml':
            logger.info("Using ML model to convert AEs to SMILES")
            topk_smiles = inference(aes2smiles_model, aes, method='beam', beam_size=args.topk, device=args.device, **configs['aes2smiles'])
            smiles_dict[aes]  = [ _.replace(' ', '') for _ in topk_smiles]
        elif args.conversion =='db':
            logger.info("Using database to convert AEs to SMILES")
            topk_smiles = mp_dbSearch(aes, args.database_dir, args.topk)
            smiles_dict[aes] = topk_smiles

    return smiles_dict


if __name__=='__main__':
    parser = argparse.ArgumentParser(description =
                                     """ Single-step retrosynthetic prediction for RetroTRAE.   \
                                     See more: https://doi.org/10.1038/s41467-022-28857-w
                                     """, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_type', default='bi',  choices=['uni', 'bi'], help="Uni-molecular or Bi-molecular model type")
    parser.add_argument('--smiles', type=str, help='An input sequence')
    parser.add_argument('--decode', type=str, default='greedy', choices=['greedy', 'beam'],  help="Decoding method for RetroTRAE")
    parser.add_argument('--beam_size', type=int, default=3, help="Beam size (a number of candidates for RetroTRAE)")
    parser.add_argument('--conversion', type=str, default='ml', choices=['ml', 'db'], help="How to convert AEs to SMILES? 'ml': Machine Learning model 'db': Retrieve from PubChem database")
    parser.add_argument('--database_dir', type=str,  default='./data/PubChem_AEs', help="Database for retrieval of the predicted molecules")
    parser.add_argument('--topk', type=int, default=1, help="A number of candidates for the AEs to SMIES conversion")
    parser.add_argument('--uni_checkpoint_name', type=str, default='uni_checkpoint.pth', help="Checkpoint file name")
    parser.add_argument('--bi_checkpoint_name', type=str,  default='bi_checkpoint.pth', help="Checkpoint file name")
    parser.add_argument('--log_file', type=str,  default=None, help="A file name for saving outputs")

    args = parser.parse_args()

    if args.log_file:
        handler = logging.FileHandler(filename=args.log_file, mode='w')
        logger.addHandler(handler)

    if not args.smiles:
        args.smiles= 'COc1cc2c(c(Cl)c1OC)CCN(C)CC2c1ccccc1'
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f"{args}")

    aes2smiles_model = utils.build_model(**configs["aes2smiles"], device=args.device)
    uni_model = utils.build_model(**configs['uni-molecular'], device=args.device)
    bi_model = utils.build_model(**configs['bi-molecular'], device=args.device)

    logger.info(f"Preprocessing input SMILES: {args.smiles}")
    input_tokens = utils.getAtomEnvs(args.smiles)
    logger.info(f"Preprocessed input tokens: {input_tokens}\n")

    logger.info(f'{"Uni molecular":*^10}')
    uni_result = predict(input_tokens, uni_model, aes2smiles_model, args, **configs['uni-molecular'])
    logger.info(f"{uni_result=}\n")

    logger.info(f'{"Bi molecular":*^10}')
    bi_result = predict(input_tokens, bi_model, aes2smiles_model,  args,  **configs['bi-molecular'])
    logger.info(f"{bi_result=}\n")
    logger.info('Done!')
