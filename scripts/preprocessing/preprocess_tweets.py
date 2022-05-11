import codecs
import json
import logging
import os
import re
import string
from argparse import ArgumentParser
from typing import Dict, List, Tuple, Set, Iterable
from ast import literal_eval
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.preprocessing.preprocessing_parameters import REPLACE_AMP_MAP, EMOJI_MAPS_MAP, VERBOSE2CLASS_ID
from scripts.preprocessing.preprocessing_utils import preprocess_tweet_text

nltk.download('punkt')


def load_drugbank_dict(json_path: str) -> Dict[str, str]:
    """
    Loads {drugname : drugbank id} mapping from file
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


def get_en_tweets_drugs(tweet_text: str, drugbank_dictionary: Dict[str, str]) -> List[Tuple[str, str, str]]:
    """
    Takes English tweet and returns its drug mentions based on Drugbank dictionary
    :param tweet_text: Tweet text string
    :param drugbank_dictionary: Dict {Drug name : Drugbank ID}
    :return: [List[Drug name, Drug name, Drugbank ID]]
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
            drugname_drugid_list.append((token, token, drug_id))
            found = True
    if not found:
        drugname_drugid_list.append((np.nan, np.nan, np.nan,))
    return drugname_drugid_list


class Drug:
    def __init__(self, en_name, name, selected_terms, compound, drug_id):
        self.en_name = en_name
        self.name = name
        self.selected_terms = selected_terms
        self.compound = compound
        self.drug_id = drug_id


def get_drugs_list(drugbank_df: pd.DataFrame, drug_name_column: str, selected_terms_column: str,
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
        selected_terms = eval(row[selected_terms_column].strip())
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


def drug_list_to_token(drug_tuples_lst, join_token, index):
    tokens_set = set([t[index] if t[index] is not np.nan else None for t in drug_tuples_lst])
    tokens_set = ['~'.join(t.split('+')).replace(' ', '') for t in tokens_set if t is not None]
    if len(tokens_set) == 0:
        return None
    return join_token.join(tokens_set)


def drug_ids_to_smiles_strings(drug_ids_str: str, id_to_smiles_mapping: Dict[str, str], drugs_sep: str = '~',
                               smiles_sep: str = '~~~') -> str:
    if drug_ids_str is not None:
        drugs_list = drug_ids_str.split(drugs_sep)
    else:
        drugs_list = []
    smiles_list = []
    for drug_id in drugs_list:
        drug_smiles = id_to_smiles_mapping[drug_id]

        if drug_smiles is not np.nan:
            smiles_list.append(drug_smiles)
    return smiles_sep.join(smiles_list)


def get_atc_codes_first_char_values_set(atc_codes_list: Iterable[str]) -> Set[str]:
    """
    Takes a list of atc codes. For each code, takes its first letter corresponding to a high-level ATC-group.
    Creates a set of there first letters
    :param atc_codes_list: List of atc codes
    :return: Set of unique first letters(each corresponds to some ATC group).
    """
    chars_set = set()
    for drug_atcs in atc_codes_list:
        for atc_code in drug_atcs:
            first_char = atc_code[0]
            chars_set.add(first_char)
    return chars_set


def get_atc_codes_by_drug_ids(data_df: pd.DataFrame, possible_atc_codes: Iterable[str],
                              id_to_atc_mapping: Dict[str, str],
                              drugs_sep: str = '~', ) -> str:
    entries = []
    for drug_ids_str in data_df["drug_id"].values:
        atc_codes_dict = {code: 0 for code in possible_atc_codes}
        if drug_ids_str is np.nan or drug_ids_str is None:
            entries.append(atc_codes_dict)
            continue
        drugs_list = re.split(rf'[+{drugs_sep}]', drug_ids_str)
        for drug_id in drugs_list:
            drug_atc_codes = id_to_atc_mapping[drug_id]
            for atc_code in drug_atc_codes:
                atc_first_char = atc_code[0]
                assert atc_first_char in possible_atc_codes
                atc_codes_dict[atc_first_char] = 1
        entries.append(atc_codes_dict)

    atc_features_df = pd.DataFrame(entries)
    return atc_features_df


def main():
    parser = ArgumentParser()

    parser.add_argument('--input_data_dir', default=r"../../data/smm4h_datasets/en_21_custom_test/raw/",
                        help="Path to the directory that contains train, dev, and test sets to be preprocessed.")
    parser.add_argument('--ru_drug_terms_path', default=r"../../data/df_all_terms_ru_en.csv", required=False,
                        help="Path to the csv file that contains the mapping between the Russian and English terms.")
    parser.add_argument('--drugbank_term2drugbank_id_path', default=r"../../data/drugbank_aliases.json", required=False,
                        help="Path to the json file that contains possible drug mentions for English and French grouped"
                             "by DrugBank id")
    parser.add_argument('--drugbank_metadata_path', default=r"../../data/drugbank_database.csv",
                        help="Path to the csv file that contains drug metadata for each Drug identified with DrugBank"
                             "id. This file contains ATC codes and SMILES strings.")
    parser.add_argument('--language', default=r"en",
                        help="Language of tweets to be preprocessed.")
    parser.add_argument('--output_dir', default=r"../../data/smm4h_datasets/en_21_custom_test/preprocessed_tweets/", )
    args = parser.parse_args()

    input_data_dir = args.input_data_dir
    language = args.language
    output_dir = args.output_dir
    if not os.path.exists(output_dir) and not output_dir == '':
        os.makedirs(output_dir)
    drugbank_metadata_path = args.drugbank_metadata_path
    drugbank_metadata_df = pd.read_csv(drugbank_metadata_path, converters={"atc_codes": literal_eval})
    drugbank_id_smiles_df = drugbank_metadata_df[["drugbank_id", "smiles"]]
    drugbank_id_smiles_df.set_index("drugbank_id", inplace=True)
    drugbank_id_smiles_df = drugbank_id_smiles_df.squeeze()
    amp_replace = REPLACE_AMP_MAP[language]
    emoji_mapping = EMOJI_MAPS_MAP[language]

    for filename in os.listdir(input_data_dir):
        dataset_type = filename.split('.')[0]
        if not dataset_type in ("train", "dev", "valid", "test"):
            raise Exception(f"Invalid filename: {filename}")
        input_tweets_path = os.path.join(input_data_dir, filename)
        output_path = os.path.join(output_dir, filename)
        tweets_df = pd.read_csv(input_tweets_path, sep='\t', quoting=3, dtype={"tweet_id": str, "user_id": str})
        # Removing duplicated tweets
        if dataset_type != "test":
            logging.info(f"Filtering duplicates in {dataset_type}. There are {tweets_df.shape[0]} in total.")
            tweets_df.drop_duplicates(inplace=True)
            tweets_df.reset_index(inplace=True, drop=True)
            logging.info(f"There are {tweets_df.shape[0]} in {dataset_type} after duplicates filtering.")
        # Finding drug mentions and mapping them to DrugBank id
        if language == "ru":
            if args.ru_drug_terms_path is None:
                raise Exception(f"Russian drug terms are required")
            ru_drug_terms_path = args.ru_drug_terms_path
            ru_drug_terms_df = pd.read_csv(ru_drug_terms_path, )
            drugbank_list = get_drugs_list(ru_drug_terms_df, drug_name_column="Drug name in Russian",
                                           selected_terms_column="selected_terms_w_cases",
                                           active_compound_column="Active compound in Russian")
            tweets_drugs = []
            for tweet_text in tweets_df.tweet.values:
                drugs_present_in_tweet = get_tweet_drug_from_list(tweet_text, drugbank_list)
                tweets_drugs.append(drugs_present_in_tweet)
        elif language in ("en", "fr"):
            if args.drugbank_term2drugbank_id_path is None:
                raise Exception(f"English/French term to drugbank id mapping is required")
            drugbank_term2drugbank_id_path = args.drugbank_term2drugbank_id_path
            drugbank_dict = load_drugbank_dict(drugbank_term2drugbank_id_path)
            tweets_drugs = []
            for tweet_text in tqdm(tweets_df.tweet.values):
                tweet_drug = get_en_tweets_drugs(tweet_text, drugbank_dict)
                tweets_drugs.append(tweet_drug)
        else:
            raise Exception(f"Unsupported language: {language}")

        drug_en_names_list = [drug_list_to_token(t, '~', 0) for t in tweets_drugs]
        drug_names_list = [drug_list_to_token(t, '~', 1) for t in tweets_drugs]
        drug_ids_list = [drug_list_to_token(t, '~', 2) for t in tweets_drugs]

        tweets_df["drug_en_name"] = drug_en_names_list
        tweets_df["drug_name"] = drug_names_list
        tweets_df["drug_id"] = drug_ids_list
        logging.info(f"{tweets_df.drug_id.isna().sum()} tweets are not matched to any drug in {dataset_type}")
        tweets_df["smiles"] = tweets_df.drug_id.apply(lambda x: drug_ids_to_smiles_strings(x, drugbank_id_smiles_df))

        # Assigning ATC codes to tweets
        drugbank_id_atc_code_df = drugbank_metadata_df[["drugbank_id", "atc_codes"]]
        drugbank_id_atc_code_df.set_index("drugbank_id", inplace=True)
        possible_atc_first_chars_set = ("A", "B", "C", "D", "G", "H", "J", "L", "M", "N", "P", "R", "S", "V")
        # possible_atc_first_chars_set = get_atc_codes_first_char_values_set(drugbank_metadata_df["atc_codes"].values)
        # possible_atc_first_chars_set = sorted(possible_atc_first_chars_set)
        drugbank_id_atc_code_df = drugbank_id_atc_code_df.squeeze()

        atc_features_df = get_atc_codes_by_drug_ids(tweets_df, possible_atc_first_chars_set, drugbank_id_atc_code_df)

        tweets_df = pd.concat((tweets_df, atc_features_df), axis=1)

        if "label" in tweets_df.columns:
            tweets_df.rename(columns={"label": "class"}, inplace=True)
        if "class" not in tweets_df.columns:
            tweets_df["class"] = 0
        tweets_df["class"] = tweets_df["class"].replace(VERBOSE2CLASS_ID)
        # Preprocessing texts: url, username, masking; emoji replacing
        tweets_df['tweet'] = tweets_df['tweet'].apply(lambda x: preprocess_tweet_text(x, emoji_mapping, amp_replace))

        tweets_df.to_csv(output_path, sep='\t', index=False, quoting=3)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    main()
