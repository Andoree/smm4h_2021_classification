import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm.notebook import tqdm

from scripts.baselines.eval_baseline import evaluate_classification


class CNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, out_channels, kernel_sizes, embedding_weights, dropout=0.5,):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim, _weight=embedding_weights)
        self.conv_0 = nn.Sequential(
            nn.Conv1d(emb_dim, out_channels, kernel_size=kernel_sizes[0], ),
            nn.BatchNorm1d(out_channels),
        )

        self.conv_1 = nn.Sequential(
            nn.Conv1d(emb_dim, out_channels, kernel_size=kernel_sizes[1], ),
            nn.BatchNorm1d(out_channels),
        )

        self.conv_2 = nn.Sequential(
            nn.Conv1d(emb_dim, out_channels, kernel_size=kernel_sizes[2], ),
            nn.BatchNorm1d(out_channels),
        )

        self.fc = nn.Linear(len(kernel_sizes) * out_channels, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.permute(0, 2, 1)

        conved_0 = F.relu(self.conv_0(embedded))
        conved_1 = F.relu(self.conv_1(embedded))
        conved_2 = F.relu(self.conv_2(embedded))

        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim=1))

        return self.fc(cat)


def train_cnn(cnn_model, loss_fn, optimizer, train_loader, val_loader, decision_threshold, num_epochs, device):
    cnn_model.train()
    best_val_f1 = -1.
    for e in tqdm(range(num_epochs)):
        train_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad()

            batch_train_output = cnn_model(input_ids).squeeze(1)
            loss = loss_fn(batch_train_output, target)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        print(f"Epoch {e + 1} train. Loss: {train_loss:.4f}")

        val_loss = 0.0
        cnn_model.eval()
        val_epoch_predictions = []
        val_epoch_true_y = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                target = batch["target"].to(device)

                batch_val_output = cnn_model(input_ids).squeeze(1)
                loss = loss_fn(batch_val_output, target)
                val_loss += loss.item()

                val_epoch_true_y.extend(target.detach().cpu().numpy().astype(int))
                batch_val_probas = batch_val_output.detach().cpu().sigmoid().numpy()
                batch_predicted_labels = (batch_val_probas >= decision_threshold).astype(int)
                val_epoch_predictions.extend(batch_predicted_labels)
        val_eval_dict = evaluate_classification(true_y=val_epoch_true_y, pred_y=val_epoch_predictions)
        val_loss /= len(val_loader)
        pr, rec, f1 = val_eval_dict["Precision"], val_eval_dict["Recall"], val_eval_dict["F1"]

        print(f"Epoch {e + 1} validation. Loss: {val_loss:.4f}, F1: {f1:.4f}, Precision: {pr:.4f}, Recall: {rec:.4f}")
        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(cnn_model.state_dict(), "best_cnn.pt")
    cnn_model.load_state_dict(torch.load("best_cnn.pt"))



class CNNTweetsDataset(Dataset):
    def __init__(self, tokenized_texts, targets, ):
        assert len(tokenized_texts) == len(targets)
        self.tokenized_texts = tokenized_texts
        self.targets = targets

    def __getitem__(self, idx):
        d = {
            "input_ids": torch.LongTensor(self.tokenized_texts[idx]),
            "target": float(self.targets[idx])
        }
        return d

    def __len__(self):
        return len(self.targets)


def cnn_predict(cnn_model, data_loader, decision_threshold, device):
    cnn_model.eval()
    predicted_y = []
    true_y = []
    with torch.no_grad():
        for batch in tqdm(data_loader, miniters=len(data_loader) // 100):
            input_ids = batch["input_ids"].to(device)
            target = batch["target"].to(device)
            batch_output = cnn_model(input_ids).squeeze(1)
            true_y.extend(target.detach().cpu().numpy().astype(int))
            batch_pred_probas = batch_output.detach().cpu().sigmoid().numpy()
            batch_predicted_labels = (batch_pred_probas >= decision_threshold).astype(int)
            predicted_y.extend(batch_predicted_labels)
    return true_y, predicted_y
