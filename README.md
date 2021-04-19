# smm4h_2021_classification


## Preprocessing

1. Preprocessing tweets text:

English tweets:
```
python3 preprocess_tweet_texts.py --input_dir data/smm4h_21_data/en/raw --lang en --output_dir data/smm4h_21_data/en/preprocessed
```
Russian tweets:
```
python3 preprocess_tweet_texts.py --input_dir data/smm4h_21_data/ru/raw --lang ru --output_dir data/smm4h_21_data/ru/preprocessed
```
2. Finding drug mentions and mapping the mentions to Drugbank:

English tweets:
```
python3 map_en_tweets_to_drugbank.py --input_tweets_path data/smm4h_21_data/en/preprocessed/train.tsv --input_drugbank_path data/drugbank_aliases.json --not_matched_path data/smm4h_21_data/en/not_matched_en_train.tsv --output_path data/smm4h_21_data/en/tweets_w_drugs/train.tsv
python3 map_en_tweets_to_drugbank.py --input_tweets_path data/smm4h_21_data/en/preprocessed/dev.tsv --input_drugbank_path data/drugbank_aliases.json --not_matched_path data/smm4h_21_data/en/not_matched_en_dev.tsv --output_path data/smm4h_21_data/en/tweets_w_drugs/dev.tsv
python3 map_en_tweets_to_drugbank.py --input_tweets_path data/smm4h_21_data/en/preprocessed/test.tsv --input_drugbank_path data/drugbank_aliases.json --not_matched_path data/smm4h_21_data/en/not_matched_en_test.tsv --output_path data/smm4h_21_data/en/tweets_w_drugs/test.tsv
```
Russian tweets:
```
python3 map_ru_tweets_to_drugbank.py --input_tweets_path data/smm4h_21_data/ru/preprocessed/train.tsv --input_drugbank_path data/df_all_terms_ru_en.csv --not_matched_path data/smm4h_21_data/ru/not_matched_ru_train.tsv --output_path data/smm4h_21_data/ru/tweets_w_drugs/train.tsv
python3 map_ru_tweets_to_drugbank.py --input_tweets_path data/smm4h_21_data/ru/preprocessed/valid.tsv --input_drugbank_path data/df_all_terms_ru_en.csv --not_matched_path data/smm4h_21_data/ru/not_matched_ru_dev.tsv --output_path data/smm4h_21_data/ru/tweets_w_drugs/dev.tsv
python3 map_ru_tweets_to_drugbank.py --input_tweets_path data/smm4h_21_data/ru/preprocessed/test.tsv --input_drugbank_path data/df_all_terms_ru_en.csv --not_matched_path data/smm4h_21_data/ru/not_matched_ru_test.tsv --output_path data/smm4h_21_data/ru/tweets_w_drugs/test.tsv
```

3. Get drug SMILES strings:

English tweets:
```
python3 get_tweets_smiles.py --input_drugbank data/drugbank_database.csv --input_tweets data/smm4h_21_data/en/tweets_w_drugs/train.tsv --output_path data/smm4h_21_data/en/tweets_w_smiles/train.tsv
python3 get_tweets_smiles.py --input_drugbank data/drugbank_database.csv --input_tweets data/smm4h_21_data/en/tweets_w_drugs/dev.tsv --output_path data/smm4h_21_data/en/tweets_w_smiles/dev.tsv
python3 get_tweets_smiles.py --input_drugbank data/drugbank_database.csv --input_tweets data/smm4h_21_data/en/tweets_w_drugs/test.tsv --output_path data/smm4h_21_data/en/tweets_w_smiles/test.tsv
```
Russian tweets

```
python3 get_tweets_smiles.py --input_drugbank data/drugbank_database.csv --input_tweets data/smm4h_21_data/ru/tweets_w_drugs/train.tsv --output_path data/smm4h_21_data/ru/tweets_w_smiles/train.tsv
python3 get_tweets_smiles.py --input_drugbank data/drugbank_database.csv --input_tweets data/smm4h_21_data/ru/tweets_w_drugs/dev.tsv --output_path data/smm4h_21_data/ru/tweets_w_smiles/dev.tsv
python3 get_tweets_smiles.py --input_drugbank data/drugbank_database.csv --input_tweets data/smm4h_21_data/ru/tweets_w_drugs/test.tsv --output_path data/smm4h_21_data/ru/tweets_w_smiles/test.tsv
```

## Training


## Ensembling & evaluation


