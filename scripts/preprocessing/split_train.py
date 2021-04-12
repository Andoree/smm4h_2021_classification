import os
from argparse import ArgumentParser

import pandas as pd


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_path', default=r"../../data/smm4h_21_data/en/tweets_w_smiles/train.tsv")
    parser.add_argument('--train_frac', type=float, default=0.9)
    parser.add_argument('--output_dir', default=r"../../data/smm4h_21_data/en/tweets_w_smiles_val_stage/")
    args = parser.parse_args()

    input_path = args.input_path
    train_frac = args.train_frac
    output_dir = args.output_dir
    if not os.path.exists(output_dir) and not output_dir == '':
        os.makedirs(output_dir)

    data_df = pd.read_csv(input_path, sep='\t', quoting=3)
    data_df = data_df.sample(frac=1.0, random_state=42)
    num_samples = data_df.shape[0]
    print(f"Num samples: {num_samples}" )
    num_train_samples = int(num_samples * train_frac)
    num_valid_samples = num_samples - num_train_samples
    print(f"Num train samples: {num_train_samples}")
    print(f"Num valid samples: {num_valid_samples}")
    train_df = data_df[:num_train_samples]
    valid_df = data_df[num_train_samples:]
    print(f"Train size: {train_df.shape}")
    print(f"Train positive: {train_df[train_df['class'] == 1].shape[0]}")
    print(f"Train negative: {train_df[train_df['class'] == 0].shape[0]}")
    print(f"Valid size: {valid_df.shape}")
    print(f"Valid positive: {valid_df[valid_df['class'] == 1].shape[0]}")
    print(f"Valid negative: {valid_df[valid_df['class'] == 0].shape[0]}")

    output_train_path = os.path.join(output_dir, "train.tsv")
    output_valid_path = os.path.join(output_dir, "dev.tsv")
    train_df.to_csv(output_train_path, sep='\t', index=False, quoting=3)
    valid_df.to_csv(output_valid_path, sep='\t', index=False, quoting=3)


if __name__ == '__main__':
    main()
