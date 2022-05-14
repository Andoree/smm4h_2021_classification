import os.path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from scripts.preprocessing.preprocessing_parameters import POSSIBLE_ATC_CODE_LETTERS


def create_atc_stratified_train_dev_test(data_df: pd.DataFrame, train_size=0.8, val_size=0.1) \
        -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,]:
    present_atc_code_letters = sorted(list(set(data_df.columns).intersection(POSSIBLE_ATC_CODE_LETTERS)))
    no_code_values_list = []
    for _, row in data_df.iterrows():
        has_code = False
        for atc_letter in present_atc_code_letters:
            if row[atc_letter] == 1:
                has_code = True
                no_code_values_list.append(0)
                break
        if not has_code:
            no_code_values_list.append(1)
    data_df["NO_CODE"] = no_code_values_list
    present_atc_code_letters.append("NO_CODE")

    tweet_ids_grouped_by_atc_code = {}
    for atc_letter in sorted(present_atc_code_letters):
        atc_subset_df = data_df[data_df[atc_letter] == 1]
        atc_subset_tweet_ids_set = set(atc_subset_df["tweet_id"].values)
        tweet_ids_grouped_by_atc_code[atc_letter] = atc_subset_tweet_ids_set
    train_subset_dfs_list = []
    val_subset_dfs_list = []
    test_subset_dfs_list = []
    seen_letters = set()
    for atc_letter in tweet_ids_grouped_by_atc_code.keys():
        atc_subset_tweet_ids_set = tweet_ids_grouped_by_atc_code[atc_letter]
        overall_num_tweets = len(atc_subset_tweet_ids_set)

        num_train_samples = int(overall_num_tweets * train_size)
        num_val_samples = int(num_train_samples * val_size)
        num_train_samples = num_train_samples - num_val_samples

        atc_subset_tweet_ids_array = np.array(list(atc_subset_tweet_ids_set))
        # Sorting tweet ids to guarantee np.random.shuffle() reproducibility
        atc_subset_tweet_ids_array.sort()
        np.random.shuffle(atc_subset_tweet_ids_array, )

        train_tweet_ids = atc_subset_tweet_ids_array[:num_train_samples]
        val_tweet_ids = atc_subset_tweet_ids_array[num_train_samples:num_train_samples + num_val_samples]
        test_tweet_ids = atc_subset_tweet_ids_array[num_train_samples + num_val_samples:]

        train_subset_df = data_df[data_df["tweet_id"].isin(train_tweet_ids)]
        val_subset_df = data_df[data_df["tweet_id"].isin(val_tweet_ids)]
        test_subset_df = data_df[data_df["tweet_id"].isin(test_tweet_ids)]

        train_subset_dfs_list.append(train_subset_df)
        val_subset_dfs_list.append(val_subset_df)
        test_subset_dfs_list.append(test_subset_df)
        seen_letters.add(atc_letter)
        for atc_letter_2 in tweet_ids_grouped_by_atc_code.keys():
            if atc_letter_2 not in seen_letters:
                atc_subset_tweet_ids_to_be_filtered = tweet_ids_grouped_by_atc_code[atc_letter_2]
                atc_subset_tweet_ids_to_be_filtered = atc_subset_tweet_ids_to_be_filtered.difference(
                    atc_subset_tweet_ids_set)
                tweet_ids_grouped_by_atc_code[atc_letter_2] = atc_subset_tweet_ids_to_be_filtered

    train_df = pd.concat(train_subset_dfs_list, axis=0).reset_index(drop=True)
    val_df = pd.concat(val_subset_dfs_list, axis=0).reset_index(drop=True)
    test_df = pd.concat(test_subset_dfs_list, axis=0).reset_index(drop=True)

    return train_df, val_df, test_df


def main():
    input_not_strat_data_dir = "../../data/smm4h_datasets/en_21_custom_test/preprocessed_tweets"
    output_stratified_data_dir = "../../data/smm4h_datasets/en_21_atc_stratified_test/preprocessed_tweets"

    input_not_strat_train_path = os.path.join(input_not_strat_data_dir, "train.tsv")
    input_not_strat_dev_path = os.path.join(input_not_strat_data_dir, "dev.tsv")
    input_not_strat_test_path = os.path.join(input_not_strat_data_dir, "test.tsv")

    out_stratified_train_path = os.path.join(output_stratified_data_dir, "train.tsv")
    out_stratified_dev_path = os.path.join(output_stratified_data_dir, "dev.tsv")
    out_stratified_test_path = os.path.join(output_stratified_data_dir, "test.tsv")

    if not os.path.exists(output_stratified_data_dir):
        os.makedirs(output_stratified_data_dir)

    np.random.seed(42)

    not_strat_train_df = pd.read_csv(input_not_strat_train_path, sep="\t", encoding="utf-8", quoting=3,
                                     dtype={"tweet_id": str})
    not_strat_dev_df = pd.read_csv(input_not_strat_dev_path, sep="\t", encoding="utf-8", quoting=3,
                                   dtype={"tweet_id": str})
    not_strat_test_df = pd.read_csv(input_not_strat_test_path, sep="\t", encoding="utf-8", quoting=3,
                                    dtype={"tweet_id": str})

    print(f"Not stratified train shape: {not_strat_train_df.shape}")
    print(f"Not stratified dev shape: {not_strat_dev_df.shape}")
    print(f"Not stratified test shape: {not_strat_test_df.shape}")
    full_data_df = pd.concat((not_strat_train_df, not_strat_dev_df, not_strat_test_df), axis=0).reset_index(drop=True)
    print(f"Full data shape: {full_data_df.shape}")
    stratified_train_df, stratified_dev_df, stratified_test_df = create_atc_stratified_train_dev_test(full_data_df, train_size=0.8, val_size=0.1)
    unique_train_tweet_ids = set(stratified_train_df.tweet_id.values)
    unique_dev_tweet_ids = set(stratified_dev_df.tweet_id.values)
    unique_test_tweet_ids = set(stratified_test_df.tweet_id.values)

    train_dev_intersection = unique_train_tweet_ids.intersection(unique_dev_tweet_ids)
    train_test_intersection = unique_train_tweet_ids.intersection(unique_test_tweet_ids)
    dev_test_intersection = unique_dev_tweet_ids.intersection(unique_test_tweet_ids)
    assert len(train_dev_intersection) == len(train_test_intersection) == len(dev_test_intersection) == 0

    stratified_train_df.drop("NO_CODE", inplace=True, axis=1)
    stratified_dev_df.drop("NO_CODE", inplace=True, axis=1)
    stratified_test_df.drop("NO_CODE", inplace=True, axis=1)

    print(f"Stratified train shape: {stratified_train_df.shape}")
    print(f"Stratified dev shape: {stratified_dev_df.shape}")
    print(f"Stratified test shape: {stratified_test_df.shape}")

    stratified_train_df.to_csv(out_stratified_train_path, sep='\t', index=False, quoting=3)
    stratified_dev_df.to_csv(out_stratified_dev_path, sep='\t', index=False, quoting=3)
    stratified_test_df.to_csv(out_stratified_test_path, sep='\t', index=False, quoting=3)


if __name__ == '__main__':
    main()
