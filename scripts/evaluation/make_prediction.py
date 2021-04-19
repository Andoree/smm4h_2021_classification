import os
from argparse import ArgumentParser

import pandas as pd


def main():
    parser = ArgumentParser()
    parser.add_argument('--prediction_tsv',
                        default=r"predictions/eval_stage/post_eval/en/roberta_attention_random_freeze5_upsampling.tsv")
    parser.add_argument('--prediction_column', default="Class")
    parser.add_argument('--lang', default="en")
    parser.add_argument('--output_path',
                        default=r"predictions/eval_stage/post_eval/en/submission_roberta_attention_random_freeze5_upsampling.tsv")
    args = parser.parse_args()

    prediction_tsv_path = args.prediction_tsv
    prediction_column = args.prediction_column
    lang = args.lang
    output_path = args.output_path
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and not output_dir == '':
        os.makedirs(output_dir)

    prediction_df = pd.read_csv(prediction_tsv_path, sep="\t", encoding="utf-8", )

    print("Prediction", prediction_df.shape)

    prediction_df.drop("tweet", axis=1, inplace=True)
    result_df = prediction_df[["tweet_id", prediction_column]]
    print("Result", result_df.shape)
    result_df.rename(columns={prediction_column: "label"}, inplace=True)
    if lang == "en":
        replace_dict = {0: "NoADE",
                        1: "ADE"}
        result_df.replace(replace_dict, inplace=True)

    result_df.to_csv(output_path, sep='\t', index=False)


if __name__ == '__main__':
    main()
