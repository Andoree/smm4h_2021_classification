import os
from argparse import ArgumentParser
from typing import List, Tuple

import numpy as np
import pandas as pd


class Drug:
    def __init__(self, en_name, name, selected_terms, compound, drug_id):
        self.en_name = en_name
        self.name = name
        self.selected_terms = selected_terms
        self.compound = compound
        self.drug_id = drug_id


def get_drugbank_list(drugbank_df: pd.Dataframe, drug_name_column: str, selected_terms_column: str,
                      active_compound_column: str) -> List[Drug]:
    """
    Converts Drugbank entries into Drug class instances
    :param drugbank_df: Drugbank dataframe
    :param drug_name_column: Drug name Dataframe column
    :param selected_terms_column: Dataframe column that contains list of probable drug mentions
    :param active_compound_column: Dataframe column of drug active compound
    :return: List of drugs from Drugbank
    """
    drugbank_list = []
    for idx, row in drugbank_df.iterrows():
        drug_name = row[drug_name_column]
        drug_en_name = row["Drug"]
        drug_id = row["drugbank_id"]
        selected_terms = [x.strip("'").lower() for x in row[selected_terms_column].strip('[]').split(', ')]

        compound = row[active_compound_column].lower()
        drug = Drug(en_name=drug_en_name, name=drug_name, selected_terms=selected_terms, drug_id=drug_id,
                    compound=compound)
        drugbank_list.append(drug)
    return drugbank_list


def get_tweet_drug_from_list(tweet_text: str, drugs_list: List[Drug]) -> List[Tuple[str, str, str]]:
    """
    :param tweet_text: Tweet text string
    :param drugs_list: List of drugs from Drugbank. We check each drug for the presence
    in target tweet.
    :return: List of tuples (Drug English name, Drug Russian name, Drug Drugbank ID)
    """
    drugname_drugid_list = []
    tweet_text = tweet_text.lower()
    found = False

    for i, drug in enumerate(drugs_list):
        if drug.name is not np.nan:
            if drug.name in tweet_text:
                drugname_drugid_list.append((drug.en_name, drug.name, drug.drug_id))
                found = True
        if drug.compound is not np.nan:
            if drug.compound in tweet_text:
                drugname_drugid_list.append((drug.en_name, drug.name, drug.drug_id))
                found = True
        for drug_selected_term in drug.selected_terms:
            if drug_selected_term in tweet_text:
                drugname_drugid_list.append((drug.en_name, drug.name, drug.drug_id))
                found = True
    if not found:
        drugname_drugid_list.append((np.nan, np.nan, np.nan))
    return drugname_drugid_list


def drug_list_to_token(lst, join_token, index):
    tokens_set = list(dict.fromkeys([t[index] if t[index] is not np.nan else None for t in lst]))
    tokens_set = ['~'.join(t.split(' + ')) for t in tokens_set if t is not None]
    if len(tokens_set) == 0:
        return None
    return join_token.join(tokens_set)


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_tweets_path', default=r"../../data/smm4h_21_data/ru/preprocessed/valid.tsv")
    parser.add_argument('--input_drugbank_path', default=r"../../df_all_terms_ru_en.csv")
    parser.add_argument('--not_matched_path', default=r"../../data/smm4h_21_data/ru/not_matched_ru_valid.tsv")
    parser.add_argument('--language', default=r"ru")
    parser.add_argument('--output_path', default=r"../../data/smm4h_21_data/ru/tweets_w_drugs/valid.tsv")
    args = parser.parse_args()

    input_tweets_path = args.input_tweets_path
    input_drugbank_path = args.input_drugbank_path
    language = args.language
    not_matched_path = args.not_matched_path
    output_dir = os.path.dirname(not_matched_path)
    if not os.path.exists(output_dir) and not output_dir == '':
        os.makedirs(output_dir)
    output_path = args.output_path
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and not output_dir == '':
        os.makedirs(output_dir)

    if language == "ru":
        drug_name_column = "Drug name in Russian"
        selected_terms_column = "selected_terms_w_cases"
        compound_column = "Active compound in Russian"
    elif language == "en":
        drug_name_column = "Drug"
        selected_terms_column = "target_med"
        compound_column = "normalized_med"
    else:
        raise ValueError(f"Invalid language: {language}")
    tweets_df = pd.read_csv(input_tweets_path, sep='\t', quoting=3, header=None, names=["class", "tweet"])

    print("Tweets before:", tweets_df.shape[0])
    tweets_df.drop_duplicates(inplace=True)
    print("Tweets after:", tweets_df.shape[0])
    drugbank_df = pd.read_csv(input_drugbank_path, )
    drugbank_list = get_drugbank_list(drugbank_df, drug_name_column=drug_name_column,
                                      selected_terms_column=selected_terms_column,
                                      active_compound_column=compound_column)

    tweets_drugs = []
    for tweet_text in tweets_df.tweet.values:
        tweet_drug = get_tweet_drug_from_list(tweet_text, drugbank_list)
        tweets_drugs.append(tweet_drug)

    drug_en_names_list = [drug_list_to_token(t, '~', 0) for t in tweets_drugs]
    drug_names_list = [drug_list_to_token(t, '~', 1) for t in tweets_drugs]
    drug_ids_list = [drug_list_to_token(t, '~', 2) for t in tweets_drugs]

    tweets_df["drug_en_name"] = drug_en_names_list
    tweets_df["drug_name"] = drug_names_list
    tweets_df["drug_id"] = drug_ids_list
    print("not mapped:", tweets_df.drug_id.isna().sum())
    nan_df = tweets_df[tweets_df.drug_en_name.isnull()]
    tweets_df = tweets_df[~tweets_df.drug_en_name.isnull()]
    tweets_df.to_csv(output_path, sep='\t', index=False, quoting=3)
    print(tweets_df[tweets_df["class"] == 1].shape)
    print(tweets_df[tweets_df["class"] == 0].shape)
    nan_df.to_csv(not_matched_path, sep='\t', index=False, quoting=3)


if __name__ == '__main__':
    main()
