import argparse
import os
from random import shuffle

import sentencepiece as spm

from parameters import *


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


def split_data(model_type, raw_data_name, data_dir, augment=True, num_augment=5):
    with open(f"{DATA_DIR}/{model_type}_{raw_data_name}", encoding="utf-8") as f:
        lines = f.readlines()

    print("\nSplitting data...")
    if augment:
        print('*'*5, f'Train set of SRC will be augmented {num_augment} times.\n')

    temp = lines[:int(train_frac * len(lines))]

    train_lines = lines[:int(train_frac * len(temp))]
    valid_lines = lines[int(train_frac * len(temp)):]
    test_lines = lines[int(train_frac * len(lines)):]

    if not os.path.isdir(f"{DATA_DIR}/{data_dir}"):
        os.mkdir(f"{DATA_DIR}/{data_dir}")

    with open(f"{DATA_DIR}/{data_dir}/{model_type}_{TRAIN_NAME}", 'w', encoding="utf-8") as f:
        for line in train_lines:
            if augment:
                for _ in range(num_augment):
                    line_list = line.strip().split()
                    shuffle(line_list)
                    shuffled_list = ' '.join(line_list)
                    f.write(shuffled_list + '\n')
            else:
                f.write(line.strip() + '\n')

    with open(f"{DATA_DIR}/{data_dir}/{model_type}_{VALID_NAME}", 'w', encoding="utf-8") as f:
        for line in valid_lines:
            f.write(line.strip() + '\n')

    with open(f"{DATA_DIR}/{data_dir}/{model_type}_{TEST_NAME}", 'w', encoding="utf-8") as f:
        for line in test_lines:
            f.write(line.strip() + '\n')

    print(f"Train/Valid/test data saved in {DATA_DIR}/{data_dir}.\n")


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='bi', type=str, help='Enter either "uni" or "bi" or "both"')
    #parser.add_argument('--augment', action='store_true', help='Data augmentation of SRC (Products)')
    #parser.add_argument('--num_augment', type=int, default=5, help='Number of data augmentation of SRC (Products)')


    args = parser.parse_args()
    if args.model_type:
        train_sp(args.model_type, src_vocab_size[args.model_type], is_src=True)
        train_sp(args.model_type, trg_vocab_size[args.model_type], is_src=False)
        split_data(args.model_type, SRC_RAW_DATA_NAME, SRC_DIR, True, 5)
        split_data(args.model_type, TRG_RAW_DATA_NAME, TRG_DIR, False, 0)

    elif args.model_type =='both':
        train_sp('uni', src_vocab_size['uni'], is_src=True)
        train_sp('uni', trg_vocab_size['uni'], is_src=False)
        split_data('uni', SRC_RAW_DATA_NAME, SRC_DIR,  True, 5)
        split_data('uni', TRG_RAW_DATA_NAME, TRG_DIR,  False, 0)

        train_sp('bi', src_vocab_size['bi'], is_src=True)
        train_sp('bi', trg_vocab_size['bi'], is_src=False)
        split_data('bi', SRC_RAW_DATA_NAME, SRC_DIR,  True, 5)
        split_data('bi', TRG_RAW_DATA_NAME, TRG_DIR,  False, 0)

