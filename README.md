# smm4h_2021_classification


## Preprocessing

1. Preprocessing tweets text:

English tweets:
```
python3 scripts/preprocessing/preprocess_tweet_texts.py --input_dir data/smm4h_21_data/en/raw --lang en --output_dir data/smm4h_21_data/en/preprocessed
```
Russian tweets:
```
python3 scripts/preprocessing/preprocess_tweet_texts.py --input_dir data/smm4h_21_data/ru/raw --lang ru --output_dir data/smm4h_21_data/ru/preprocessed
```
2. Finding drug mentions and mapping the mentions to Drugbank:

English tweets:
```
python3 scripts/preprocessing/map_en_tweets_to_drugbank.py --input_tweets_path data/smm4h_21_data/en/preprocessed/train.tsv --input_drugbank_path data/drugbank_aliases.json --not_matched_path data/smm4h_21_data/en/not_matched_en_train.tsv --output_path data/smm4h_21_data/en/tweets_w_drugs/train.tsv
python3 scripts/preprocessing/map_en_tweets_to_drugbank.py --input_tweets_path data/smm4h_21_data/en/preprocessed/dev.tsv --input_drugbank_path data/drugbank_aliases.json --not_matched_path data/smm4h_21_data/en/not_matched_en_dev.tsv --output_path data/smm4h_21_data/en/tweets_w_drugs/dev.tsv
python3 scripts/preprocessing/map_en_tweets_to_drugbank.py --input_tweets_path data/smm4h_21_data/en/preprocessed/test.tsv --input_drugbank_path data/drugbank_aliases.json --not_matched_path data/smm4h_21_data/en/not_matched_en_test.tsv --output_path data/smm4h_21_data/en/tweets_w_drugs/test.tsv
```
Russian tweets:
```
python3 scripts/preprocessing/map_ru_tweets_to_drugbank.py --input_tweets_path data/smm4h_21_data/ru/preprocessed/train.tsv --input_drugbank_path data/df_all_terms_ru_en.csv --not_matched_path data/smm4h_21_data/ru/not_matched_ru_train.tsv --output_path data/smm4h_21_data/ru/tweets_w_drugs/train.tsv
python3 scripts/preprocessing/map_ru_tweets_to_drugbank.py --input_tweets_path data/smm4h_21_data/ru/preprocessed/valid.tsv --input_drugbank_path data/df_all_terms_ru_en.csv --not_matched_path data/smm4h_21_data/ru/not_matched_ru_dev.tsv --output_path data/smm4h_21_data/ru/tweets_w_drugs/dev.tsv
python3 scripts/preprocessing/map_ru_tweets_to_drugbank.py --input_tweets_path data/smm4h_21_data/ru/preprocessed/test.tsv --input_drugbank_path data/df_all_terms_ru_en.csv --not_matched_path data/smm4h_21_data/ru/not_matched_ru_test.tsv --output_path data/smm4h_21_data/ru/tweets_w_drugs/test.tsv
```

3. Getting drug SMILES strings:

English tweets:
```
python3 scripts/preprocessing/get_tweets_smiles.py --input_drugbank data/drugbank_database.csv --input_tweets data/smm4h_21_data/en/tweets_w_drugs/train.tsv --output_path data/smm4h_21_data/en/tweets_w_smiles/train.tsv
python3 scripts/preprocessing/get_tweets_smiles.py --input_drugbank data/drugbank_database.csv --input_tweets data/smm4h_21_data/en/tweets_w_drugs/dev.tsv --output_path data/smm4h_21_data/en/tweets_w_smiles/dev.tsv
python3 scripts/preprocessing/get_tweets_smiles.py --input_drugbank data/drugbank_database.csv --input_tweets data/smm4h_21_data/en/tweets_w_drugs/test.tsv --output_path data/smm4h_21_data/en/tweets_w_smiles/test.tsv
```
Russian tweets

```
python3 scripts/preprocessing/get_tweets_smiles.py --input_drugbank data/drugbank_database.csv --input_tweets data/smm4h_21_data/ru/tweets_w_drugs/train.tsv --output_path data/smm4h_21_data/ru/tweets_w_smiles/train.tsv
python3 scripts/preprocessing/get_tweets_smiles.py --input_drugbank data/drugbank_database.csv --input_tweets data/smm4h_21_data/ru/tweets_w_drugs/dev.tsv --output_path data/smm4h_21_data/ru/tweets_w_smiles/dev.tsv
python3 scripts/preprocessing/get_tweets_smiles.py --input_drugbank data/drugbank_database.csv --input_tweets data/smm4h_21_data/ru/tweets_w_drugs/test.tsv --output_path data/smm4h_21_data/ru/tweets_w_smiles/test.tsv
```

4. Combining the Russian and English tweets sets:
```
python3 scripts/preprocessing/merge_tweets_sets.py --input_files data/smm4h_21_data/ru/tweets_w_smiles/train.tsv data/smm4h_21_data/en/tweets_w_smiles/train.tsv --output_path data/smm4h_21_data/ruen/tweets_w_smiles/train.tsv
python3 scripts/preprocessing/merge_tweets_sets.py --input_files data/smm4h_21_data/ru/tweets_w_smiles/dev.tsv data/smm4h_21_data/en/tweets_w_smiles/dev.tsv --output_path data/smm4h_21_data/ruen/tweets_w_smiles/dev.tsv
python3 scripts/preprocessing/merge_tweets_sets.py --input_files data/smm4h_21_data/ru/tweets_w_smiles/test.tsv data/smm4h_21_data/en/tweets_w_smiles/test.tsv --output_path data/smm4h_21_data/ruen/tweets_w_smiles/test.tsv
```


## Training

Training hyperparameters (scripts/training/train_config.ini):

	[INPUT]
  
	INPUT_DIR - Dataset directory that contains train.tsv, dev.tsv and test.tsv files
  
	DRUG_EMBEDDINGS_FROM - Drug embeddings source. "chemberta" setup loads HuggingFace's ChemBERTa model and encodes drugs dynamically.
  
	[PARAMETERS]
  
	SEED - random state
  
	MAX_TEXT_LENGTH - Text encoder maximum sequence length
  
	MAX_MOLECULE_LENGTH - Drug encoder maximum sequence length
  
	BATCH_SIZE - Batch size
  
	LEARNING_RATE - Training learning rate
  
	DROPOUT - Dense classification layer dropout probability
  
	NUM_EPOCHS - Maximum number of training epochs
  
	APPLY_UPSAMPLING - Boolean. Whether to use positive class oversampling
  
	USE_WEIGHTED_LOSS - Boolean. Whether to use weighted loss to increase the positive class weight

	LOSS_WEIGHT - Positive class weight. if set to '-1', class weights will be proportional to theirs inverted frequencies
  
	MODEL_TYPE - Model architecture
  
	TEXT_ENCODER_NAME - Text encoder model HuggingFace's name.
  
	DRUG_SAMPLING - Drug sampling type. "random": During training, a drug from each sample is drawn at random when there are multiple drug mentions. "first" : The first found drug is used.
  
	[UPSAMPLING]
  
	UPSAMPLING_WEIGHT - Upsampling weight. It is used if APPLY_UPSAMPLING is set to True. if set to '-1', class sampling probabilities will be proportional to theirs inverted frequencies
  
	[CROSSATT_PARAM] - Drug-text Cross-attention architecture hyperparameters
  
	CROSSATT_DROPOUT - Cross-attention layer token-level dropout probability
  
	CROSSATT_HIDDEN_DROPOUT - Cross-attention hidden layer dropour probability

	[OUTPUT]
  
	OUTPUT_DIR - Output directory. Contains model checkpoint, evaluation results, predicted labeld and probabilities
  
	EVALUATION_FILENAME - Evaluation filename

Training script:

```
python3 scripts/training/train_drug_text_bert.py
```

## Ensembling & evaluation


