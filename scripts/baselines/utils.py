import os
from typing import List, Dict

import fasttext
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors


def load_data(input_data_dir: str):
    train_path = os.path.join(input_data_dir, "train.tsv")
    dev_path = os.path.join(input_data_dir, "dev.tsv")
    test_path = os.path.join(input_data_dir, "test.tsv")

    train_df = pd.read_csv(train_path, sep="\t", encoding="utf-8", quoting=3, )
    dev_df = pd.read_csv(dev_path, sep="\t", encoding="utf-8", quoting=3, )
    test_df = pd.read_csv(test_path, sep="\t", encoding="utf-8", quoting=3, )

    return train_df, dev_df, test_df


def list_replace(search, replacement, text):
    """
    Replaces all symbols of text which are present
    in the search string with the replacement string.
    """
    search = [el for el in search if el in text]
    for c in search:
        text = text.replace(c, replacement)
    return text


def create_embedding_matrix(word_index, embeddings_model, emb_dim):
    dictionary_size = len(word_index.keys())
    # 0-th token of embedding matrix is a padding token
    embedding_matrix = np.zeros((dictionary_size, emb_dim), dtype=float)

    for word, i in word_index.items():
        if isinstance(embeddings_model, fasttext.FastText._FastText):
            embedding_vector = embeddings_model.get_word_vector(word)
        elif isinstance(embeddings_model, KeyedVectors):
            if word in embeddings_model:
                embedding_vector = embeddings_model.get_vector(word, norm=True)
            else:
                embedding_vector = np.zeros(shape=embeddings_model.vector_size, dtype=float)
        else:
            raise Exception(f"Unsupported embedding model class: {type(embeddings_model)}")
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def pad_texts(texts_list: List[List[str]], token2id: Dict[str, int], embeddings_model_vocab, max_text_length: int):
    padded_texts = []
    for tokens_list in texts_list:
        if embeddings_model_vocab is not None:
            tokens_list = [token for token in tokens_list if token in embeddings_model_vocab]

        # tokens_list = tokens_list + ["<pad>", ] * (max_text_length - text_length)
        token_ids = [token2id[token.strip().lower()] for token in tokens_list
                     if token2id.get(token.strip().lower()) is not None]
        text_length = len(token_ids)
        token_ids = token_ids + [token2id["<pad>"], ] * max(0, (max_text_length - text_length))
        token_ids = token_ids[:max_text_length]
        padded_texts.append(token_ids)
    return padded_texts


def tokenize_pad_texts(train_texts: List[List[str]], dev_texts: List[List[str]],
                       test_texts: List[List[str]], embeddings_model, max_text_length: int):
    tokens_set = set()
    all_texts = train_texts + dev_texts + test_texts
    for tokens_list in all_texts:
        tokens_set.update(tokens_list)
    embeddings_model_vocab = None
    if isinstance(embeddings_model, KeyedVectors):
        embeddings_model_vocab = set(embeddings_model.key_to_index.keys())
        tokens_set = set(
            (token.strip().lower() for token in tokens_set if token.strip().lower() in embeddings_model_vocab))
    elif isinstance(embeddings_model, fasttext.FastText._FastText):
        tokens_set = set(
            (token.strip().lower() for token in tokens_set if
             embeddings_model.get_word_vector(token.strip().lower()) is not None))

    token2id = {token.strip().lower(): idx for idx, token in enumerate(tokens_set)}

    num_tokens = len(token2id.keys())
    token2id["<pad>"] = num_tokens
    # Converting texts to lists of ids and padding too short texts
    train_padded_token_ids = pad_texts(texts_list=train_texts, token2id=token2id,
                                       embeddings_model_vocab=embeddings_model_vocab, max_text_length=max_text_length)
    dev_padded_token_ids = pad_texts(texts_list=dev_texts, token2id=token2id,
                                     embeddings_model_vocab=embeddings_model_vocab, max_text_length=max_text_length)
    test_padded_token_ids = pad_texts(texts_list=test_texts, token2id=token2id,
                                      embeddings_model_vocab=embeddings_model_vocab, max_text_length=max_text_length)

    return train_padded_token_ids, dev_padded_token_ids, test_padded_token_ids, token2id
