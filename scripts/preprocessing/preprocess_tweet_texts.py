import os
from argparse import ArgumentParser

import pandas as pd
from preprocessing_parameters import REPLACE_AMP_MAP, EMOJI_MAPS_MAP
from preprocessing_utils import preprocess_tweet_text


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_dir', default=r"../../data/smm4h_21_data/en/raw")
    parser.add_argument('--lang', default="ru")
    parser.add_argument('--output_dir', default=r"../../data/smm4h_21_data/en/preprocessed")
    args = parser.parse_args()

    input_dir = args.input_dir
    language = args.lang
    output_dir = args.output_dir
    if not os.path.exists(output_dir) and not output_dir == '':
        os.makedirs(output_dir)
    amp_replace = REPLACE_AMP_MAP[language]
    emoji_mapping = EMOJI_MAPS_MAP[language]

    for filename in os.listdir(input_dir):
        dataset_type = filename.split('.')[0]
        if dataset_type == 'train' or dataset_type == 'valid' or dataset_type == 'dev':
            columns = ["label", "tweet"]
        elif dataset_type == 'test':
            if language == "ru":
                columns = ["tweet_id", "tweet"]
            else:
                columns = ["tweet_id", "tweet"]
        else:
            raise Exception(f"Invalid filename: {filename}")

        input_path = os.path.join(input_dir, filename)
        data_df = pd.read_csv(input_path, sep="\t", encoding="utf-8", quoting=3)[columns]
        replace_map = {
            "NoADE": 0,
            "ADE": 1
        }
        data_df["label"] = data_df["label"].replace(replace_map)
        if dataset_type == "test" and language == "ru":
            data_df.rename(columns={"label": "class"}, inplace=True)
        data_df['tweet'] = data_df['tweet'].apply(lambda x: preprocess_tweet_text(x, emoji_mapping, amp_replace))
        output_path = os.path.join(output_dir, filename)
        data_df.to_csv(output_path, encoding="UTF-8", sep="\t", index=False, header=None, quoting=3)


if __name__ == '__main__':
    main()
