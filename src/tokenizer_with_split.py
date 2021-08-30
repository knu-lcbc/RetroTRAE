from parameters import *
from tqdm import tqdm

import os, argparse
import sentencepiece as spm

train_frac = 0.90

def train_sp(model_type, vocab_size, is_src=True):
    template = "--input={} \
                --pad_id={} \
                --bos_id={} \
                --eos_id={} \
                --unk_id={} \
                --model_prefix={} \
                --vocab_size={} \
                --character_coverage={} \
                --model_type={}"


    if is_src:
        this_input_file = f"{DATA_DIR}/{model_type}_{SRC_RAW_DATA_NAME}"
        this_model_prefix = f"{SP_DIR}/{model_type}_{src_model_prefix}"

        config = template.format(this_input_file,
                                pad_id,
                                sos_id,
                                eos_id,
                                unk_id,
                                this_model_prefix,
                                vocab_size,
                                character_coverage,
                                sp_model_type)

        print(config)

        if not os.path.isdir(SP_DIR):
            os.mkdir(SP_DIR)

        print(spm)
        spm.SentencePieceTrainer.Train(config)



    else:
        this_input_file = f"{DATA_DIR}/{model_type}_{TRG_RAW_DATA_NAME}"
        this_model_prefix = f"{SP_DIR}/{model_type}_{trg_model_prefix}"

        config = template.format(this_input_file,
                                pad_id,
                                sos_id,
                                eos_id,
                                unk_id,
                                this_model_prefix,
                                vocab_size,
                                character_coverage,
                                sp_model_type)

        print(config)

        if not os.path.isdir(SP_DIR):
            os.mkdir(SP_DIR)

        print(spm)
        spm.SentencePieceTrainer.Train(config)


def split_data(model_type, raw_data_name, data_dir):
    with open(f"{DATA_DIR}/{model_type}_{raw_data_name}", encoding="utf-8") as f:
        lines = f.readlines()

    print("Splitting data...")

    train_lines = lines[:int(train_frac * len(lines))]
    valid_lines = lines[int(train_frac * len(lines)):]

    if not os.path.isdir(f"{DATA_DIR}/{data_dir}"):
        os.mkdir(f"{DATA_DIR}/{data_dir}")

    with open(f"{DATA_DIR}/{data_dir}/{model_type}_{TRAIN_NAME}", 'w', encoding="utf-8") as f:
        for line in tqdm(train_lines):
            f.write(line.strip() + '\n')

    with open(f"{DATA_DIR}/{data_dir}/{model_type}_{VALID_NAME}", 'w', encoding="utf-8") as f:
        for line in tqdm(valid_lines):
            f.write(line.strip() + '\n')

    print(f"Train/Validation data saved in {DATA_DIR}/{data_dir}.")


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True, help='Enter either "uni" or "bi"')

    args = parser.parse_args()
    train_sp(args.model_type, src_vocab_size[args.model_type], is_src=True)
    train_sp(args.model_type, trg_vocab_size[args.model_type], is_src=False)
    split_data(args.model_type, SRC_RAW_DATA_NAME, SRC_DIR)
    split_data(args.model_type, TRG_RAW_DATA_NAME, TRG_DIR)

