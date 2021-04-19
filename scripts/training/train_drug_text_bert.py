import codecs
import configparser
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from attention import BertCrossattLayer
from sklearn.metrics import precision_score, f1_score, recall_score
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModel, RobertaModel
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available else "cpu"


class TweetsDataset(Dataset):
    def __init__(self, tweets_df, text_tokenizer, molecule_tokenizer=None, molecule_max_length=256,
                 text_max_length=128, sampling_type="first", ):
        self.labels = tweets_df["class"].astype(np.float32).values
        self.text_max_length = text_max_length
        self.sampling_type = sampling_type
        self.molecule_max_length = molecule_max_length
        self.tokenized_tweets = [text_tokenizer.encode_plus(x, max_length=self.text_max_length,
                                                            padding="max_length", truncation=True,
                                                            return_tensors="pt", ) for x in tweets_df.tweet.values]
        self.tokenized_molecules = None
        if molecule_tokenizer is not None:
            smiles_list = get_smiles_list(tweets_df.smiles.values)
            self.tokenized_molecules = [molecule_tokenizer.batch_encode_plus(x, max_length=self.molecule_max_length,
                                                                             padding="max_length", truncation=True,
                                                                             return_tensors="pt", ) for x in
                                        smiles_list]
        self.drug_embeddings = tweets_df.drug_embedding.values
        self.smiles = tweets_df.smiles.values

    def __getitem__(self, idx):
        sample_dict = {
            "input_ids": self.tokenized_tweets[idx]["input_ids"][0],
            "attention_mask": self.tokenized_tweets[idx]["attention_mask"][0],
            "drug_embeddings": self.drug_embeddings[idx],
            "labels": self.labels[idx]
        }
        if self.tokenized_molecules is not None:
            if self.sampling_type == "random":
                num_samples = self.tokenized_molecules[idx]["input_ids"].size()[0]
                sample_id = random.randint(0, num_samples - 1)
                sample_dict["molecule_input_ids"] = self.tokenized_molecules[idx]["input_ids"][sample_id]
                sample_dict["molecule_attention_mask"] = self.tokenized_molecules[idx]["attention_mask"][sample_id]
            else:
                sample_dict["molecule_input_ids"] = self.tokenized_molecules[idx]["input_ids"][0]
                sample_dict["molecule_attention_mask"] = self.tokenized_molecules[idx]["attention_mask"][0]
        return sample_dict

    def __len__(self):
        return len(self.labels)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def create_dataset_weights(dataset, positive_class_weight=-1.0):
    count_dict = {}
    for item in dataset:
        label = item["labels"]
        if count_dict.get(label) is None:
            count_dict[label] = 0
        count_dict[label] += 1
    num_samples = len(dataset)
    label_to_weight = {}
    assert num_samples == sum(count_dict.values())
    count_0 = count_dict[0]
    count_1 = count_dict[1]
    freq_0 = count_0 / num_samples
    freq_1 = count_1 / num_samples
    label_to_weight[0] = 1 - freq_0
    if positive_class_weight <= 0:
        label_to_weight[1] = 1 - freq_1
    else:
        label_to_weight[1] = label_to_weight[0] * positive_class_weight
    sample_weights = np.empty(num_samples, dtype=np.float)
    for i, item in enumerate(dataset):
        label = item["labels"]
        sample_weights[i] = label_to_weight[label]
    return sample_weights


def train(model, iterator, optimizer, criterion, use_drug_embeddings=True, cross_att_flag=False, ):
    model.train()

    epoch_loss = 0
    history = []
    for i, batch in enumerate(iterator):

        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        assert not (cross_att_flag and use_drug_embeddings)
        if use_drug_embeddings:
            drug_embeddings = batch["drug_embeddings"].to(device)
            output = model(inputs=input_ids, attention_mask=attention_mask, drug_embeddings=drug_embeddings, ).squeeze(
                1)
        elif cross_att_flag:
            molecule_input_ids = batch["molecule_input_ids"].to(device)
            molecule_attention_mask = batch["molecule_attention_mask"].to(device)
            output = model(text_inputs=input_ids, text_attention_mask=attention_mask,
                           molecule_inputs=molecule_input_ids,
                           molecule_attention_mask=molecule_attention_mask).squeeze(1)
        else:
            output = model(inputs=input_ids, attention_mask=attention_mask, ).squeeze(1)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        history.append(loss.cpu().data.numpy())

    return epoch_loss / (i + 1)


def get_smiles_list(smiles_list, molecules_sep='~~~'):
    preprocessed_smiles = []
    for smile_str in smiles_list:
        if smile_str is np.nan:
            preprocessed_smiles.append([""])
        else:
            preprocessed_smiles.append(smile_str.split(molecules_sep))
    return preprocessed_smiles


def encode_smiles(model, tokenizer, smiles_list, max_length, molecules_sep='~~~'):
    model.eval()
    with torch.no_grad():
        model_hidden_size = model.config.hidden_size
        molecules_embeddings = []
        for sample in tqdm(smiles_list):
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


def evaluate(model, iterator, criterion, use_drug_embeddings, cross_att_flag=False, ):
    model.eval()
    epoch_loss = 0
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            true_labels.extend(labels.cpu().numpy())
            labels = labels.to(device)

            assert not (cross_att_flag and use_drug_embeddings)
            if use_drug_embeddings:
                drug_embeddings = batch["drug_embeddings"].to(device)
                output = model(inputs=input_ids, attention_mask=attention_mask,
                               drug_embeddings=drug_embeddings, ).squeeze(1)
            elif cross_att_flag:
                molecule_input_ids = batch["molecule_input_ids"].to(device)
                molecule_attention_mask = batch["molecule_attention_mask"].to(device)
                output = model(text_inputs=input_ids, text_attention_mask=attention_mask,
                               molecule_inputs=molecule_input_ids,
                               molecule_attention_mask=molecule_attention_mask).squeeze(1)
            else:
                output = model(inputs=input_ids, attention_mask=attention_mask, ).squeeze(1)
            pred_probas = output.cpu().numpy()
            batch_pred_labels = (pred_probas >= 0.5) * 1

            loss = criterion(output, labels)

            pred_labels.extend(batch_pred_labels)
            epoch_loss += loss.item()

    valid_f1_score = f1_score(true_labels, pred_labels)
    return epoch_loss / (i + 1), valid_f1_score


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_evaluate(bert_classifier, train_loader, dev_loader, optimizer, criterion, n_epochs, use_drug_embeddings,
                   save_checkpoint_path, output_evaluation_path, cross_att_flag=False, ):
    train_history = []
    valid_history = []
    valid_history_f1 = []

    best_f1_score = 0.0
    best_epoch = -1

    eval_dir = os.path.dirname(output_evaluation_path)
    train_statistics_path = os.path.join(eval_dir, "training_logs.txt")

    for epoch in tqdm(range(n_epochs)):

        start_time = time.time()

        train_loss = train(bert_classifier, train_loader, optimizer, criterion, use_drug_embeddings,
                           cross_att_flag=cross_att_flag, )
        valid_loss, valid_f1_score = evaluate(bert_classifier, dev_loader, criterion, use_drug_embeddings,
                                              cross_att_flag=cross_att_flag, )

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        train_history.append(train_loss)
        valid_history.append(valid_loss)
        valid_history_f1.append(valid_f1_score)

        if valid_f1_score > best_f1_score:
            best_f1_score = valid_f1_score
            best_epoch = epoch
            torch.save(bert_classifier.state_dict(), save_checkpoint_path)

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. F1: {valid_f1_score:.3f}')

        with codecs.open(train_statistics_path, 'a+', encoding="utf-8") as output_path:
            output_path.write(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s\n')
            output_path.write(f'\tTrain Loss: {train_loss:.3f}\n')
            output_path.write(f'\t Val. Loss: {valid_loss:.3f} |  Val. F1: {valid_f1_score:.3f}\n')

    return best_epoch


def save_labels_probas(labels_path, probas_path, labels, probas):
    with codecs.open(labels_path, 'w+', encoding="utf-8") as labels_file, \
            codecs.open(probas_path, 'w+', encoding="utf-8") as probas_file:
        for label, probability in zip(labels, probas):
            labels_file.write(f"{label}\n")
            probas_file.write(f"{probability}\n")


def train_evaluate_model(seed, bert_classifier, use_drug_embeddings, criterion, learning_rate, train_loader, dev_loader,
                         test_loader, num_epochs, output_evaluation_path, output_model_dir, model_chkpnt_name,
                         cross_att_flag=False, ):
    torch.manual_seed(seed)
    optimizer = optim.Adam(bert_classifier.parameters(), lr=learning_rate)
    output_ckpt_path = os.path.join(output_model_dir, f"best-val-{model_chkpnt_name}.pt")
    best_epoch = train_evaluate(bert_classifier, train_loader, dev_loader, optimizer, criterion, num_epochs,
                                use_drug_embeddings, output_ckpt_path, output_evaluation_path,
                                cross_att_flag=cross_att_flag, )

    bert_classifier.load_state_dict(torch.load(output_ckpt_path))

    true_labels, pred_labels, pred_probas = predict(bert_classifier, train_loader, use_drug_embeddings,
                                                    cross_att_flag=cross_att_flag, )
    save_labels_probas(labels_path=os.path.join(output_model_dir, "pred_train_labels.txt"),
                       probas_path=os.path.join(output_model_dir, "pred_train_probas.txt"), labels=pred_labels,
                       probas=pred_probas)
    true_labels, pred_labels, pred_probas = predict(bert_classifier, dev_loader, use_drug_embeddings,
                                                    cross_att_flag=cross_att_flag, )
    assert len(pred_labels) == len(pred_probas)
    assert len(true_labels) == len(pred_labels)
    val_model_precision = precision_score(true_labels, pred_labels)
    val_model_recall = recall_score(true_labels, pred_labels)
    val_model_f1 = f1_score(true_labels, pred_labels)
    save_labels_probas(labels_path=os.path.join(output_model_dir, "pred_dev_labels.txt"),
                       probas_path=os.path.join(output_model_dir, "pred_dev_probas.txt"), labels=pred_labels,
                       probas=pred_probas)

    true_labels, pred_labels, pred_probas = predict(bert_classifier, test_loader, use_drug_embeddings,
                                                    cross_att_flag=cross_att_flag, )
    assert len(pred_labels) == len(pred_probas)
    assert len(true_labels) == len(pred_labels)
    test_model_precision = precision_score(true_labels, pred_labels)
    test_model_recall = recall_score(true_labels, pred_labels)
    test_model_f1 = f1_score(true_labels, pred_labels)
    save_labels_probas(labels_path=os.path.join(output_model_dir, "pred_test_labels.txt"),
                       probas_path=os.path.join(output_model_dir, "pred_test_probas.txt"), labels=pred_labels,
                       probas=pred_probas)

    with codecs.open(output_evaluation_path, 'a+', encoding="utf-8") as output_file:
        output_file.write(f"{model_chkpnt_name},{best_epoch},{val_model_precision},{val_model_recall},{val_model_f1}\n")
        output_file.write(
            f"{model_chkpnt_name},{best_epoch},{test_model_precision},{test_model_recall},{test_model_f1}\n")

    del optimizer
    del criterion


def predict(model, data_loader, use_drug_embeddings, cross_att_flag=False, decision_threshold=0.5, ):
    true_labels = []
    pred_labels = []
    pred_probas = []

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_true_labels = batch["labels"].cpu().numpy()
            assert not (cross_att_flag and use_drug_embeddings)
            if use_drug_embeddings:
                drug_embeddings = batch["drug_embeddings"].to(device)
                batch_pred_probas = model(inputs=input_ids, attention_mask=attention_mask,
                                          drug_embeddings=drug_embeddings, ).squeeze(1)
            elif cross_att_flag:
                molecule_input_ids = batch["molecule_input_ids"].to(device)
                molecule_attention_mask = batch["molecule_attention_mask"].to(device)
                batch_pred_probas = model(text_inputs=input_ids, text_attention_mask=attention_mask,
                                          molecule_inputs=molecule_input_ids,
                                          molecule_attention_mask=molecule_attention_mask).squeeze(1)
            else:
                batch_pred_probas = model(inputs=input_ids, attention_mask=attention_mask, ).squeeze(1)

            batch_pred_probas = batch_pred_probas.cpu().numpy()

            batch_pred_labels = (batch_pred_probas >= decision_threshold) * 1

            pred_labels.extend(batch_pred_labels)
            true_labels.extend(batch_true_labels)
            pred_probas.extend(batch_pred_probas)
    return true_labels, pred_labels, pred_probas


class BertSimpleClassifier(nn.Module):
    def __init__(self, bert_text_encoder, dropout, ):
        super().__init__()

        self.bert_text_encoder = bert_text_encoder
        bert_hidden_dim = bert_text_encoder.config.hidden_size
        self.emb_dropout = nn.Dropout(p=dropout)
        classifier_input_size = bert_hidden_dim
        self.classifier = nn.Sequential(
            nn.GELU(),
            nn.Linear(classifier_input_size, bert_hidden_dim),
            nn.Dropout(p=dropout),
            nn.GELU(),
            nn.Linear(bert_hidden_dim, 1),
        )

    def forward(self, inputs, attention_mask, ):
        last_hidden_states = self.bert_text_encoder(inputs, attention_mask=attention_mask,
                                                    return_dict=True)['last_hidden_state']
        text_cls_embeddings = torch.stack([elem[0, :] for elem in last_hidden_states])
        text_cls_embeddings = self.emb_dropout(text_cls_embeddings)
        proba = self.classifier(text_cls_embeddings)
        return proba


class BertClassifierWithDrugEmbeddings(nn.Module):
    def __init__(self, bert_text_encoder, drug_enc_hid_dim, dropout, ):
        super().__init__()

        self.bert_text_encoder = bert_text_encoder
        bert_hidden_dim = bert_text_encoder.config.hidden_size
        self.emb_dropout = nn.Dropout(p=dropout)
        classifier_input_size = bert_hidden_dim + drug_enc_hid_dim

        self.classifier = nn.Sequential(
            nn.GELU(),
            nn.Linear(classifier_input_size, bert_hidden_dim),
            nn.Dropout(p=dropout),
            nn.GELU(),
            nn.Linear(bert_hidden_dim, 1),
        )

    def forward(self, inputs, attention_mask, drug_embeddings, ):
        last_hidden_states = self.bert_text_encoder(inputs, attention_mask=attention_mask,
                                                    return_dict=True)['last_hidden_state']
        text_cls_embeddings = torch.stack([elem[0, :] for elem in last_hidden_states])
        text_cls_embeddings = self.emb_dropout(text_cls_embeddings)
        concat_text_drug_embeddings = torch.cat([text_cls_embeddings, drug_embeddings], dim=1)

        proba = self.classifier(concat_text_drug_embeddings)
        return proba


class CrossModalityBertClassifier(nn.Module):
    def __init__(self, bert_text_encoder, bert_molecule_encoder, classifier_dropout, cross_att_attention_dropout,
                 cross_att_hidden_dropout, ):
        super().__init__()

        self.bert_text_encoder = bert_text_encoder
        self.bert_molecule_encoder = bert_molecule_encoder
        text_bert_hidden_dim = bert_text_encoder.config.hidden_size
        molecule_bert_hidden_dim = bert_molecule_encoder.config.hidden_size
        num_attention_heads = text_bert_hidden_dim // 64
        self.cross_attention_layer = BertCrossattLayer(text_bert_hidden_dim, molecule_bert_hidden_dim,
                                                       cross_att_attention_dropout, cross_att_hidden_dropout,
                                                       num_attention_heads=num_attention_heads)
        classifier_input_size = text_bert_hidden_dim
        self.classifier = nn.Sequential(
            nn.Dropout(p=classifier_dropout),
            nn.GELU(),
            nn.Linear(classifier_input_size, text_bert_hidden_dim),
            nn.Dropout(p=classifier_dropout),
            nn.GELU(),
            nn.Linear(text_bert_hidden_dim, 1),
        )

    def forward(self, text_inputs, text_attention_mask, molecule_inputs, molecule_attention_mask, ):
        text_last_hidden_states = self.bert_text_encoder(text_inputs, attention_mask=text_attention_mask,
                                                         return_dict=True)['last_hidden_state']
        molecule_last_hidden_states = \
            self.bert_molecule_encoder(molecule_inputs, attention_mask=molecule_attention_mask,
                                       return_dict=True)['last_hidden_state']
        cross_attention_output = self.cross_attention_layer(input_tensor=text_last_hidden_states,
                                                            ctx_tensor=molecule_last_hidden_states, )
        cross_att_output_cls_embs = torch.stack([elem[0, :] for elem in cross_attention_output])
        proba = self.classifier(cross_att_output_cls_embs)
        return proba


def clear():
    os.system('cls')


def embedding_str_to_numpy(s):
    numbers_strs = s.strip("[]").split()
    emb_size = len(numbers_strs)
    embedding = np.empty(shape=emb_size, dtype=np.float)
    for i in range(emb_size):
        embedding[i] = np.float(numbers_strs[i])
    return embedding


def get_positive_class_loss_weight(data_df, class_column="class"):
    class_counts = data_df[class_column].value_counts()
    positive_class_weight = class_counts[0] / class_counts[1]
    return positive_class_weight


def main():
    config = configparser.ConfigParser()
    config.read("train_config.ini")
    seed = config.getint("PARAMETERS", "SEED")
    max_length = config.getint("PARAMETERS", "MAX_TEXT_LENGTH")
    max_chemberta_length = config.getint("PARAMETERS", "MAX_MOLECULE_LENGTH")
    batch_size = config.getint("PARAMETERS", "BATCH_SIZE")
    learning_rate = config.getfloat("PARAMETERS", "LEARNING_RATE")
    dropout_p = config.getfloat("PARAMETERS", "DROPOUT")
    num_epochs = config.getint("PARAMETERS", "NUM_EPOCHS")
    text_encoder_name = config.get("PARAMETERS", "TEXT_ENCODER_NAME")
    apply_upsampling = config.getboolean("PARAMETERS", "APPLY_UPSAMPLING")
    use_weighted_loss = config.getboolean("PARAMETERS", "USE_WEIGHTED_LOSS")
    loss_weight = config.getfloat("PARAMETERS", "LOSS_WEIGHT")
    model_type = config["PARAMETERS"]["MODEL_TYPE"]
    train_drug_sampling_type = config["PARAMETERS"]["DRUG_SAMPLING"]
    output_dir = config["OUTPUT"]["OUTPUT_DIR"]
    output_dir = os.path.join(output_dir, f"seed_{seed}")
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    output_evaluation_filename = config["OUTPUT"]["EVALUATION_FILENAME"]
    output_evaluation_path = os.path.join(output_dir, output_evaluation_filename)
    if apply_upsampling and use_weighted_loss:
        raise AssertionError(f"You can use only either weighted loss or upsampling")
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.random.manual_seed(seed)
    torch.cuda.random.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    data_dir = config["INPUT"]["INPUT_DIR"]

    train_path = os.path.join(data_dir, "train.tsv")
    test_path = os.path.join(data_dir, "test.tsv")
    dev_path = os.path.join(data_dir, "dev.tsv")
    train_df = pd.read_csv(train_path, sep='\t', )
    dev_df = pd.read_csv(dev_path, sep='\t', )
    test_df = pd.read_csv(test_path, sep='\t', )

    chemberta_model = RobertaModel.from_pretrained("seyonec/ChemBERTa_zinc250k_v2_40k", cache_dir="models/").to(
        device)
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa_zinc250k_v2_40k", cache_dir="models/")
    drug_enc_hid_dim = chemberta_model.config.hidden_size
    train_df["drug_embedding"] = encode_smiles(model=chemberta_model, tokenizer=tokenizer,
                                               smiles_list=train_df.smiles.values,
                                               max_length=max_chemberta_length, )
    dev_df["drug_embedding"] = encode_smiles(model=chemberta_model, tokenizer=tokenizer,
                                             smiles_list=dev_df.smiles.values, max_length=max_chemberta_length, )
    test_df["drug_embedding"] = encode_smiles(model=chemberta_model, tokenizer=tokenizer,
                                              smiles_list=test_df.smiles.values, max_length=max_chemberta_length, )
    chemberta_model = chemberta_model.cpu()
    del chemberta_model

    print(
        f"Datasets sizes: train {train_df.shape[0]},\n"
        f"dev: {dev_df.shape[0]},\n"
        f"test: {test_df.shape[0]}")
    bert_text_encoder = AutoModel.from_pretrained(text_encoder_name, cache_dir="models/")

    print("#Trainable params: ", sum(p.numel() for p in bert_text_encoder.parameters() if p.requires_grad))

    text_tokenizer = AutoTokenizer.from_pretrained(text_encoder_name, cache_dir="models/")

    if model_type == "attention" or model_type == "concat":
        chemberta_tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa_zinc250k_v2_40k", cache_dir="models/")
    else:
        chemberta_tokenizer = None

    train_tweets_dataset = TweetsDataset(train_df, text_tokenizer, text_max_length=max_length,
                                         molecule_tokenizer=chemberta_tokenizer, sampling_type=train_drug_sampling_type)
    dev_tweets_dataset = TweetsDataset(dev_df, text_tokenizer, text_max_length=max_length,
                                       molecule_tokenizer=chemberta_tokenizer, )
    test_tweets_dataset = TweetsDataset(test_df, text_tokenizer, text_max_length=max_length,
                                        molecule_tokenizer=chemberta_tokenizer, )

    if apply_upsampling:
        positive_class_weight = config.getfloat("UPSAMPLING", "UPSAMPLING_WEIGHT")
        train_weights = create_dataset_weights(train_tweets_dataset, positive_class_weight)
        print("Sampling weights:", set(train_weights))
        train_weights = torch.DoubleTensor(train_weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))
        shuffle = False
    else:
        sampler = None
        shuffle = True

    num_workers = 4

    train_loader = torch.utils.data.DataLoader(
        train_tweets_dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler, shuffle=shuffle,
        drop_last=True,
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_tweets_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_tweets_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False,
    )
    use_drug_embeddings = False
    cross_att_flag = False
    if use_weighted_loss:
        if loss_weight < 0:
            pos_weight = [get_positive_class_loss_weight(train_df, ), ]
        else:
            pos_weight = [loss_weight, ]
        print("pos_weight", pos_weight)
        pos_weight = torch.FloatTensor(pos_weight).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    else:
        criterion = nn.BCEWithLogitsLoss().to(device)
    if model_type == "nodrug":
        torch.manual_seed(seed)
        use_drug_embeddings = False
        bert_classifier = BertSimpleClassifier(bert_text_encoder, dropout=dropout_p, ).to(device)
        checkpoint_name = f"simple_{text_encoder_name.split('/')[-1]}"
    elif model_type == "drug":
        torch.manual_seed(seed)
        use_drug_embeddings = True
        bert_classifier = BertClassifierWithDrugEmbeddings(bert_text_encoder, drug_enc_hid_dim=drug_enc_hid_dim,
                                                           dropout=dropout_p, ).to(device)
        checkpoint_name = f"drug_{text_encoder_name.split('/')[-1]}"
    elif model_type == "attention":
        cross_att_attention_dropout = config.getfloat("CROSSATT_PARAM", "CROSSATT_DROPOUT")
        cross_att_hidden_dropout = config.getfloat("CROSSATT_PARAM", "CROSSATT_HIDDEN_DROPOUT")
        chemberta_model = RobertaModel.from_pretrained("seyonec/ChemBERTa_zinc250k_v2_40k", cache_dir="models/").to(
            device)
        use_drug_embeddings = False
        cross_att_flag = True
        bert_classifier = CrossModalityBertClassifier(bert_text_encoder=bert_text_encoder,
                                                      bert_molecule_encoder=chemberta_model,
                                                      classifier_dropout=dropout_p,
                                                      cross_att_attention_dropout=cross_att_attention_dropout,
                                                      cross_att_hidden_dropout=cross_att_hidden_dropout).to(device)
        checkpoint_name = f"attention_{text_encoder_name.split('/')[-1]}"
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    train_evaluate_model(seed, bert_classifier, use_drug_embeddings, criterion, learning_rate, train_loader,
                         dev_loader, test_loader, num_epochs, output_evaluation_path, output_dir, checkpoint_name,
                         cross_att_flag=cross_att_flag, )

    del bert_classifier
    del bert_text_encoder


if __name__ == '__main__':
    main()
