from typing import List, Set
import spacy


def create_spacy_lemmatization_fn(spacy_model_name):
    def lemmatize_texts(texts: List[str]):
        lemmatizer = spacy.load(spacy_model_name)
        lemmatized_texts = []
        for text in texts:
            doc = lemmatizer(text)
            lemm_txt = " ".join([_.lemma_ for _ in doc if _.pos_ != "PUNCT"])
            lemmatized_texts.append(lemm_txt)
        return lemmatized_texts

    return lemmatize_texts


def create_spacy_tokenization_fn(spacy_model_name: str, stopwords: Set[str], lemmatize: bool):
    def tokenize_texts(texts: List[str]) -> List[List[str]]:
        lemmatizer = spacy.load(spacy_model_name)
        preprocessed_texts = []
        for text in texts:
            doc = lemmatizer(text)
            if lemmatize:
                preprocessed_tokens_list = [_.lemma_ for _ in doc if _.pos_ != "PUNCT" and _.lemma_ not in stopwords]
            else:
                preprocessed_tokens_list = [_.text for _ in doc if _.pos_ != "PUNCT" and _.lemma_ not in stopwords]
            preprocessed_texts.append(preprocessed_tokens_list)
        return preprocessed_texts

    return tokenize_texts


