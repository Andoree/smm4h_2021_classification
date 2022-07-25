# smm4h_2021_classification




## Preprocessing

1. Extract data archive:
```
unzip data.zip

cd data/

unzip  drugbank_aliases_and_metadata.zip
```
drugbank_aliases_and_metadata.zip archive contains the mappings from DrugBank IDs to precomputed drug embeddings.

2. Preprocessing tweets text:

SMM4H 2021 English tweets (original test set):
```
python preprocess_tweets.py --input_data_dir="../../data/smm4h_datasets/en_21/raw/" \
--ru_drug_terms_path="../../data/df_all_terms_ru_en.csv" \
--drugbank_term2drugbank_id_path="../../data/drugbank_aliases.json" \
--drugbank_metadata_path="../../data/drugbank_database.csv" \
--language="en" \
--output_dir="../../data/smm4h_datasets/en_21/preprocessed_tweets/"
```

SMM4H 2021 English tweets (Custom test set obtained by splitting the original train into train and test sets):
```
python preprocess_tweets.py --input_data_dir="../../data/smm4h_datasets/en_21_dev_as_test/raw/" \
--ru_drug_terms_path="../../data/df_all_terms_ru_en.csv" \
--drugbank_term2drugbank_id_path="../../data/drugbank_aliases.json" \
--drugbank_metadata_path="../../data/drugbank_database.csv" \
--language="en" \
--output_dir="../../data/smm4h_datasets/en_21_dev_as_test/preprocessed_tweets/"
```

SMM4H 2020 French tweets:
```
python preprocess_tweets.py --input_data_dir="../../data/smm4h_datasets/fr_20/raw/" \
--ru_drug_terms_path="../../data/df_all_terms_ru_en.csv" \
--drugbank_term2drugbank_id_path="../../data/drugbank_aliases.json" \
--drugbank_metadata_path="../../data/drugbank_database.csv" \
--language="fr" \
--output_dir="../../data/smm4h_datasets/fr_20/preprocessed_tweets/"
```

SMM4H 2021 Russian tweets:
```
python preprocess_tweets.py --input_data_dir="../../data/smm4h_datasets/ru_21/raw/" \
--ru_drug_terms_path="../../data/df_all_terms_ru_en.csv" \
--drugbank_term2drugbank_id_path="../../data/drugbank_aliases.json" \
--drugbank_metadata_path="../../data/drugbank_database.csv" \
--language="ru" \
--output_dir="../../data/smm4h_datasets/ru_21/preprocessed_tweets/"
```


3. Combining the Russian and English tweets sets:
```
python3 scripts/preprocessing/merge_tweets_sets.py --input_files data/smm4h_21_data/ru/tweets_w_smiles/train.tsv data/smm4h_21_data/en/tweets_w_smiles/train.tsv --output_path data/smm4h_21_data/ruen/tweets_w_smiles/train.tsv
python3 scripts/preprocessing/merge_tweets_sets.py --input_files data/smm4h_21_data/ru/tweets_w_smiles/dev.tsv data/smm4h_21_data/en/tweets_w_smiles/dev.tsv --output_path data/smm4h_21_data/ruen/tweets_w_smiles/dev.tsv
python3 scripts/preprocessing/merge_tweets_sets.py --input_files data/smm4h_21_data/ru/tweets_w_smiles/test.tsv data/smm4h_21_data/en/tweets_w_smiles/test.tsv --output_path data/smm4h_21_data/ruen/tweets_w_smiles/test.tsv
```


## Training

#### SMM4H competition

For the official participation in the SMM4H 2021 Shared task, we used a script with the following set of hyperparameters:

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

4. Model training (SMM4H 2021 Shared task version):

```
python3 scripts/training/train_drug_text_bert_competition.py
```

####  Post-SMM4H training script 

For our experiments after the SMM4H 2021 competition, we used a slightly modified script ('scripts/training/train_drug_text_bert_post_competition.py').


Training in the SMM4H 2020 french using bi-modal attention-based model
```
python train_drug_text_bert_post_competition.py --input_data_dir="../../data/smm4h_datasets/fr_20/preprocessed_tweets/" \
--num_epochs 10 \
--max_length=128 \
--batch_size=64 \
--learning_rate=3e-5 \
--apply_upsampling \
--upsampling_weight 10.0 \
--freeze_layer_count 5 \
--freeze_embeddings_layer \
--text_encoder_name camembert-base \
--model_type attention \
--drug_sampling_type random \
--drug_features_path="../../data/drug_features/drug_features/chemberta_features.txt" \
--output_dir="results/smm4h_2020/attention"
```


## Ensembling & evaluation

5. Majority voting:

```
python3 scripts/evaluation/majority_voting.py --predicted_probs_dir $pred_probas \
--data_tsv data/smm4h_21_data/ru/tweets_w_smiles/test.tsv \
--probas_fname pred_test_probas.txt \
--threshold 0.5 \
--output_path prediction.tsv
```

6. Making two-column submission file:

```
python3 scripts/evaluation/make_prediction.py --prediction_tsv prediction.tsv \
--prediction_column Class \
--lang en \
--output_path submission_prediction.tsv 
```

## Environment

The code is tested with Python 3.8 and Torch 1.8.1. For more details on versions of Python libraries please refer to requirements.txt.


