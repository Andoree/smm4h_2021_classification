import os
from argparse import ArgumentParser

import pandas as pd


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_files', nargs='+', default=[r"../../data/smm4h_21_data/post_eval/ru/train.tsv",
                                                             r"../../data/smm4h_21_data/en/evaluation_stage_w_smiles/train.tsv"])
    parser.add_argument('--output_path', default=r"../../data/smm4h_21_data/post_eval/ruen/train.tsv")
    args = parser.parse_args()

    random_state = 42
    input_paths_list = args.input_files
    output_path = args.output_path
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and not output_dir == '':
        os.makedirs(output_dir)

    dataframes_list = []
    for input_path in input_paths_list:
        columns = ["class", "tweet", "drug_en_name", "drug_id", "smiles"]
        data_df = pd.read_csv(input_path, sep="\t", encoding="utf-8", )[columns]
        print(data_df.shape)
        dataframes_list.append(data_df)
    result_df = pd.concat(dataframes_list).sample(frac=1, random_state=random_state)
    print("Result shape", result_df.shape)

    result_df.to_csv(output_path, encoding="UTF-8", sep="\t", index=False, )


if __name__ == '__main__':
    main()
