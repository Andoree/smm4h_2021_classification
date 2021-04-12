import codecs
import json
import os
import string
from argparse import ArgumentParser
from typing import Dict, List, Tuple

import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.preprocessing.map_ru_tweets_to_drugbank import drug_list_to_token


def load_drugbank_dict(json_path: str) -> Dict[str, str]:
    """
    Loads drugname - drugbank id mapping from file
    :param json_path: Drugbank dict file
    :return: Dict {Drug name : Drugbank ID}
    """
    drugname_to_drugbank_id = {}
    with codecs.open(json_path, 'r', encoding="utf-8") as inp_file:
        data = json.load(inp_file)
        for drug_id, drugnames_list in data.items():
            for drugname in drugnames_list:
                drugname_to_drugbank_id[drugname.lower()] = drug_id
    return drugname_to_drugbank_id


def get_en_tweets_drugs(tweet_text: str, drugbank_dictionary: Dict[str, str]) -> List[Tuple[str, str]]:
    """
    Takes English tweet and returns its drug mentions based on Drugbank dictionary
    :param tweet_text: Tweet text string
    :param drugbank_dictionary: Dict {Drug name : Drugbank ID}
    :return: [List[Drug name, Drugbank ID]]
    """
    drugname_drugid_list = []
    found = False
    alphabet = list \
            (
            '\n абвгдеёзжийклмнопрстуфхцчшщьыъэюяАБВГДЕЁЗЖИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯ0123456789§"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ')

    alphabet.append("'")
    alphabet = set(alphabet)
    alphabet = alphabet.difference(set(string.punctuation))
    cleaned_text = [sym if sym in alphabet else ' ' for sym in tweet_text]
    tweet_text = ''.join(cleaned_text)
    tokens_list = nltk.word_tokenize(tweet_text)

    for token in tokens_list:
        token = token.lower()
        drug_id = drugbank_dictionary.get(token)
        if drug_id is not None:
            drugname_drugid_list.append((token, drug_id))
            found = True
    if not found:
        drugname_drugid_list.append((np.nan, np.nan,))
    return drugname_drugid_list


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_tweets_path', default=r"../../data/smm4h_21_data/en/preprocessed/valid.tsv")
    parser.add_argument('--input_drugbank_path', default=r"../../data/drugbank_aliases.json")
    parser.add_argument('--not_matched_path', default=r"../../data/smm4h_21_data/en/not_matched_en_valid.tsv")
    parser.add_argument('--output_path', default=r"../../data/smm4h_21_data/en/tweets_w_drugs/dev.tsv")
    args = parser.parse_args()

    input_tweets_path = args.input_tweets_path
    input_drugbank_path = args.input_drugbank_path
    not_matched_path = args.not_matched_path
    output_dir = os.path.dirname(not_matched_path)
    if not os.path.exists(output_dir) and not output_dir == '':
        os.makedirs(output_dir)
    output_path = args.output_path
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and not output_dir == '':
        os.makedirs(output_dir)
    tweets_df = pd.read_csv(input_tweets_path, sep='\t', header=None, names=["class", "tweet"], quoting=3)
    print("Tweets before duplicates drop:", tweets_df.shape[0])
    tweets_df.drop_duplicates(inplace=True)
    print("Tweets after duplicates drop:", tweets_df.shape[0])

    drugbank_dict = load_drugbank_dict(input_drugbank_path)
    tweets_drugs = []
    for tweet_text in tqdm(tweets_df.tweet.values):
        tweet_drug = get_en_tweets_drugs(tweet_text, drugbank_dict)
        tweets_drugs.append(tweet_drug)

    drug_names_list = [drug_list_to_token(t, '~', 0) for t in tweets_drugs]
    drug_ids_list = [drug_list_to_token(t, '~', 1) for t in tweets_drugs]

    tweets_df["drug_en_name"] = drug_names_list
    tweets_df["drug_id"] = drug_ids_list
    print("Not mapped tweets:", tweets_df.drug_id.isna().sum())
    nan_df = tweets_df[tweets_df.drug_id.isnull()]
    tweets_df = tweets_df[~tweets_df.drug_id.isnull()]
    tweets_df.to_csv(output_path, sep='\t', index=False, quoting=3)
    nan_df.to_csv(not_matched_path, sep='\t', index=False, quoting=3)


if __name__ == '__main__':
    main()
