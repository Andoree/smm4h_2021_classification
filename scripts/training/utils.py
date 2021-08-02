import codecs
import random
import re
from random import randrange
from typing import Set, Dict, List, Tuple

import nltk
import numpy as np
import pandas as pd
import torch
from natasha import Segmenter, Doc
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available else "cpu"


def mask_drug(text: str, drugs_set: Set[str], drug_mask: str = "DRUG"):
    """
    :param text: Raw tweet string
    :param drugs_set: Set of possible forms of drug mentions
    :param drug_mask: Mask to replace drug mentions with
    :return: Tweet string with masked drug mentions
    """
    ru_letters = set("абвгдеёжзийклмнопрстуфхцчъыьэюя")
    en_letters = set('abcdefghijklmnopqrstuvwxyz')
    ru_counter = 0
    en_counter = 0
    for char in text:
        if char in ru_letters:
            ru_counter += 1
        elif char in en_letters:
            en_counter += 1
    if ru_counter > en_counter:
        segmenter = Segmenter()
        natasha_doc = Doc(text)
        natasha_doc.segment(segmenter)
        tokens = [token.text for token in natasha_doc.tokens]
    else:
        tokens = nltk.word_tokenize(text)

    replace_tokens = []
    for token in tokens:
        if token.lower() in drugs_set:
            replace_tokens.append(token)
    replace_tokens.sort(key=lambda t: -len(t), )
    for token in replace_tokens:
        text = re.sub(token, drug_mask, text, flags=re.IGNORECASE)

    return text


def get_smiles_list(smiles_list, molecules_sep='~~~'):
    preprocessed_smiles = []
    for smile_str in smiles_list:
        if smile_str is np.nan:
            preprocessed_smiles.append([""])
        else:
            preprocessed_smiles.append(smile_str.split(molecules_sep))
    return preprocessed_smiles


def epoch_time(start_time, end_time):
    """
    Given an epoch start and end time, calculates the duration on the epoch
    :param start_time:
    :param end_time:
    :return:
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def save_labels_probas(labels_path: str, probas_path: str, labels: List[int], probas: List[float]):
    """
    Saves predicted labels and probabilities (logits) to two separate files
    :param labels_path: Path to predicted labels output file
    :param probas_path: Path to predicted probabilities (logits) output file
    :param labels: List of predicted labels
    :param probas: List of predicted positive class probabilities (or logits)
    """
    with codecs.open(labels_path, 'w+', encoding="utf-8") as labels_file, \
            codecs.open(probas_path, 'w+', encoding="utf-8") as probas_file:
        for label, probability in zip(labels, probas):
            labels_file.write(f"{label}\n")
            probas_file.write(f"{probability}\n")


def write_hyperparams(args, output_path: str):
    """
    Given config object, logs experiment hyperparameter set-up to log file
    :param args: Config object
    :param output_path: Path to save hyperparameters
    """
    with codecs.open(output_path, 'w+', encoding="utf-8") as out_file:
        for key, value in vars(args).items():
            out_file.write(f"{key} : {value}\n")
        out_file.flush()


def encode_smiles(model, tokenizer, smiles_list, max_length, molecules_sep='~~~') -> List[torch.Tensor]:
    """
    Takes a list of SMILES strings and converts converts them into embedding representation
    using a pretrained Transformer-based molecular encoder. For samples with multiple molecules,
    the mean embedding is calculated.
    :param model: pretrained molecular encoder
    :param tokenizer: Tokenizer of the pretrained encoder
    :param smiles_list: List of SMILES strings. Each element of the list consists of multiple SMILES strings
    separated with `molecules_sep`. Thus, each element of the list contains >= 1 SMILES string
    :param max_length: Maximum molecule's SMILES sequence length. Too long sequences are truncated to maximum length.
    :param molecules_sep: Separator of SMILES representations inside an element of `smiles_list`
    :return: List of mean molecular embeddings of a text sample. Each embedding is a PyTorch Tensor
    """
    model.eval()
    with torch.no_grad():
        model_hidden_size = model.config.hidden_size
        molecules_embeddings = []
        for sample in tqdm(smiles_list, mininterval=7.0):
            sample_embeddings = []
            if sample is not np.nan:
                molecules_smiles = sample.split(molecules_sep)
                for smile_str in molecules_smiles:
                    encoded_molecule = tokenizer.encode(smile_str, max_length=max_length,
                                                        padding="max_length", truncation=True, return_tensors="pt").to(
                        device)
                    output = model(encoded_molecule, return_dict=True)
                    cls_embedding = output["last_hidden_state"][0][0].cpu()
                    sample_embeddings.append(cls_embedding)
                mean_sample_embedding = torch.mean(torch.stack(sample_embeddings), dim=0)
            else:
                mean_sample_embedding = torch.zeros(size=[model_hidden_size, ], dtype=torch.float32)
            molecules_embeddings.append(mean_sample_embedding)
    return molecules_embeddings


def load_drugs_dict(dict_path: str) -> Set[str]:
    """
    Loads a set of drug names from file
    :param dict_path: Path to drug names file. Each line contains exactly one drug name
    :return: Set of unique drug names
    """
    drugs = set()
    with codecs.open(dict_path, 'r', encoding="utf-8") as inp_file:
        for line in inp_file:
            drugs.add(line.strip())
    return drugs


def load_drug_features(drug_features_path: str) -> Dict[str, np.array]:
    """
    Loads drug features from file in tsv format.
    :param drug_features_path: Path to drug features file. Each string of the file has the following structure:
    <Drugbank drug id>\t<Drug vector elements separated by whitespaces>
    :return: Dictionary {Drugbank id : Drug embedding}
    """
    drug_features_dict = {}
    with codecs.open(drug_features_path, 'r', encoding="utf-8") as inp_file:
        for line in inp_file:
            attrs = line.strip().split('\t')
            drugbank_id = attrs[0]
            feature_vector = [float(x) for x in attrs[1].split()]
            feature_vector = np.array(feature_vector, dtype=np.float32)
            drug_features_dict[drugbank_id] = feature_vector
    return drug_features_dict


def split_drugs_ids_str(drugbank_ids_str: str) -> List[str]:
    """
    Splits a string that contains multiple drug ids and returns a list of drug ids
    :param drugbank_ids_str: String that contains drug ids separated by either '+' or '~'
    :return: List of drug ids. Each drug id is a string
    """
    if drugbank_ids_str is np.nan:
        return []
    drug_ids_str = re.split(rf'[+~]', drugbank_ids_str)
    return drug_ids_str


def is_vector_zeros(vector: np.array) -> bool:
    """
    Checks whether an input vector has a non-zero element
    """
    for value in vector:
        if value != 0.0:
            return False
    return True


def sample_drug_features(drug_features_dict: Dict[str, np.array], drug_features_size: int,
                         drug_ids_list: List[str], sampling_type: str) -> np.array:
    """
    Given a dict of drug features, list of a sample's drug ids, calculates the vector representation of this sample
    :param drug_features_dict: Dict {drug id : drug vector}
    :param drug_features_size: Drug vector size
    :param drug_ids_list: List of drug ids of a sample
    :param sampling_type: Drug features sampling strategy. Can be either 'mean', 'sum', or 'random'.
    If 'random' strategy is chosen, than the drug with zero embeddings are excluded from sampling
    :return: A sample's drug vector representation
    """
    num_drugs = len(drug_ids_list)
    if num_drugs > 0:
        if sampling_type != "mean" and sampling_type != "sum":
            if sampling_type == "random":
                perm = np.random.permutation(num_drugs)
            else:
                perm = range(num_drugs)
            for i in perm:
                drug_id = drug_ids_list[i]
                drug_features = drug_features_dict.get(drug_id)
                if drug_features is not None:
                    if not is_vector_zeros(drug_features):
                        return drug_features
        else:
            drug_features_embs = []
            for drug_id in drug_ids_list:
                drug_features = drug_features_dict.get(drug_id)
                if drug_features is not None and not is_vector_zeros(drug_features):
                    drug_features_embs.append(drug_features)
            if len(drug_features_embs) > 0:
                drug_features_embs = np.array(drug_features_embs)
                if sampling_type == "mean":
                    drug_features = drug_features_embs.mean(axis=0)
                elif sampling_type == "sum":
                    drug_features = drug_features_embs.sum(axis=0)
                else:
                    raise ValueError(f"Invalid sampling type: {sampling_type}")
                return drug_features

    drug_features = np.zeros(shape=drug_features_size, dtype=np.float32)
    return drug_features


def get_drug_text_emb(text: str, drug_mention_emb_dict: Dict[str, np.array], sampling_type, drug_features_size: int):
    """
    Given a dict of drug mention embeddings, list of a sample's drug mentions calculates the vector
    representation of this sample using the pre-computed textual embeddings of drug mentions
    :param text: Sample's text. First, it is used to determine the language of the text. Second, the text is tokenized
    and the tokens are used to find drug mentions using a dictionary of drug mentions
    :param drug_mention_emb_dict: Dict {drug mention string : textual drug mention embedding}
    :param sampling_type: Drug features sampling strategy. Can be either 'mean', 'sum', or 'random'.
    If 'random' strategy is chosen, than the drug with zero embeddings are excluded from sampling
    :param drug_features_size: Drug vector size
    :return: A sample's text-based vector representation
    """
    ru_letters = set("абвгдеёжзийклмнопрстуфхцчъыьэюя")
    en_letters = set('abcdefghijklmnopqrstuvwxyz')
    fr_letters = set("abcdefghijklmnopqrstuvwxyzéèàùâêîôûëïüÿç")
    ru_counter = 0
    en_counter = 0
    fr_counter = 0
    for char in text:
        if char in ru_letters:
            ru_counter += 1
        if char in en_letters:
            en_counter += 1
        if char in fr_letters:
            fr_counter += 1
    if ru_counter > en_counter and ru_counter > fr_counter:
        segmenter = Segmenter()
        natasha_doc = Doc(text)
        natasha_doc.segment(segmenter)
        tokens = [token.text for token in natasha_doc.tokens]
    elif en_counter >= ru_counter and en_counter >= fr_counter:
        tokens = nltk.word_tokenize(text)
    elif fr_counter > ru_counter and fr_counter > en_counter:
        tokens = nltk.word_tokenize(text, language='french')
    else:
        raise Exception(f"Could not determine the language of the text: {text}")

    drugs_text_mentions = []
    drugs_set = set(list(drug_mention_emb_dict.keys()))
    for token in tokens:
        if token.lower() in drugs_set:
            drugs_text_mentions.append(token.lower())
    # drugs_text_mentions = np.array(drugs_text_mentions)
    num_drug_mentions = len(drugs_text_mentions)
    if num_drug_mentions > 0:
        if sampling_type != "mean" and sampling_type != "sum":
            if sampling_type == "random":
                sampled_mention_id = randrange(num_drug_mentions)
                sampled_mention_text = drugs_text_mentions[sampled_mention_id]
            else:
                sampled_mention_text = drugs_text_mentions[0]
            drug_text_emb = drug_mention_emb_dict[sampled_mention_text]
        else:
            drugs_text_mentions_embs = [drug_mention_emb_dict[token] for token in drugs_text_mentions if
                                        not is_vector_zeros(drug_mention_emb_dict[token])]
            drugs_text_mentions_embs = np.array(drugs_text_mentions_embs)
            if sampling_type == "mean":
                drug_text_emb = drugs_text_mentions_embs.mean(axis=0)
            elif sampling_type == "sum":
                drug_text_emb = drugs_text_mentions_embs.sum(axis=0)
            else:
                raise ValueError(f"Invalid sampling type: {sampling_type}")
    else:
        drug_text_emb = np.zeros(shape=drug_features_size, dtype=np.float32)

    return drug_text_emb


def encode_drug_text_mentions(drugs_strs: Set[str], max_seq_length: int, text_encoder, text_tokenizer) \
        -> Dict[str, np.array]:
    """
    Encodes a drug's textual name into an embedding using a pretrained textual Transformer
    :param drugs_strs:
    :param max_seq_length: Maximum sequence length. Too long sequences are truncated to maximum length.
    :param text_encoder: Pretrained textual encoder
    :param text_tokenizer: Tokenizer of the pretrained textual encoder
    :return: Textual embedding of the drug mention
    """
    drug_str_emb_dict = {}
    text_encoder.eval()
    with torch.no_grad():
        for drug_mention in drugs_strs:
            tokenizer_output = text_tokenizer.encode(drug_mention, max_length=max_seq_length,
                                                     padding="max_length", truncation=True, return_tensors="pt").to(
                device)
            output = text_encoder(tokenizer_output, return_dict=True)
            cls_embedding = output["last_hidden_state"][0][0].cpu().numpy()
            drug_str_emb_dict[drug_mention] = cls_embedding
    return drug_str_emb_dict


def create_drug_id_tweet_ids_dict(data_df: pd.DataFrame) -> Tuple[Dict[str, List[int]], np.array]:
    drug_id_tweet_ids_dict = {}
    nan_drug_tweet_ids = []
    for tweet_id, row in data_df.iterrows():
        drug_ids_str = row["drug_id"]
        if drug_ids_str is not np.nan:
            drug_ids_list = re.split(rf'[+~]', drug_ids_str)
            for drug_id in drug_ids_list:
                if drug_id_tweet_ids_dict.get(drug_id) is None:
                    drug_id_tweet_ids_dict[drug_id] = []
                drug_id_tweet_ids_dict[drug_id].append(tweet_id)
        else:
            nan_drug_tweet_ids.append(tweet_id)
    nan_drug_tweet_ids = np.array(nan_drug_tweet_ids)

    return drug_id_tweet_ids_dict, nan_drug_tweet_ids


def get_data_subset(data_df: pd.DataFrame, drug_id_tweet_ids_dict: Dict[str, List[int]], nan_tweet_ids: List[int],
                    sample_size: int):
    selected_tweet_ids = []
    num_elems_to_sample = sample_size
    while len(drug_id_tweet_ids_dict.keys()) > 0 and num_elems_to_sample > 0:
        drug_ids_list = list(drug_id_tweet_ids_dict.keys())
        num_unique_drug_ids = len(drug_ids_list)
        # Generating drug_ids sampling order
        drugs_iteration_random_order = random.sample(range(0, num_unique_drug_ids), num_unique_drug_ids)
        for i in drugs_iteration_random_order:
            drug_id = drug_ids_list[i]
            candidate_tweets_ids = drug_id_tweet_ids_dict[drug_id]
            num_candidates = len(candidate_tweets_ids)
            if num_candidates > 0:
                # Sampling a random tweet of the given drug_id
                sampled_candidate_id = random.randrange(num_candidates)
                sampled_tweet_id = candidate_tweets_ids[sampled_candidate_id]

                selected_tweet_ids.append(sampled_tweet_id)
                sampled_tweet_drug_ids_list = re.split(rf'[+~]', data_df.iloc[sampled_tweet_id]["drug_id"])
                # Removing sampled tweet's id from other drug_ids' entries
                for d_i in sampled_tweet_drug_ids_list:
                    tweet_ids_list = drug_id_tweet_ids_dict[d_i]
                    tweet_ids_list.remove(sampled_tweet_id)

                num_elems_to_sample -= 1
                if num_elems_to_sample == 0:
                    break
        for d_i in drug_ids_list:
            tweet_ids_list = drug_id_tweet_ids_dict[d_i]
            if len(tweet_ids_list) == 0:
                del drug_id_tweet_ids_dict[d_i]
    if num_elems_to_sample > 0:
        nan_sample_size = min(num_elems_to_sample, len(nan_tweet_ids))
        sampled_nan_ids = random.sample(range(0, nan_sample_size), nan_sample_size)
        selected_nan_tweets = nan_tweet_ids[sampled_nan_ids]
        selected_tweet_ids.extend(selected_nan_tweets)
    sampled_data_df = data_df.iloc[selected_tweet_ids]
    return sampled_data_df


def inclusive_range(start, stop, step):
    """
    Python's standard range(start, stop, step),
    but it always returns the right border (stop)
    """
    for i in range(start, stop, step, ):
        yield i
    if i != stop:
        yield stop
