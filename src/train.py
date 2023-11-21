import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from models.main_model import Waveformer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

epoch_accuracy = []
epoch_loss = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EEGDataset(Dataset):
    def __init__(self, data_files, label_files, window_size=32, step_size=1):
        self.data = []
        self.labels = []

        for data_file, label_file in zip(data_files, label_files):
            data = np.load(data_file)
            labels = np.load(label_file)

            for start in range(0, len(data) - window_size + 1, step_size):
                self.data.append(data[start:start + window_size])
                self.labels.append(labels[start:start + window_size])

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'data': torch.tensor(self.data[idx], dtype=torch.float),
                  'labels': torch.tensor(self.labels[idx], dtype=torch.long)}
        return sample

def train(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        X = batch['data'].to(device)
        y = batch['labels'].to(device)

        output = model(X)  # 输出形状：[batch_size, seq_len, num_classes]
        output = output.view(-1, output.shape[-1])  # 重塑为 [batch_size * seq_len, num_classes]
        y = y.view(-1)  # 重塑为 [batch_size * seq_len]

        loss = loss_fn(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        epoch_loss.append(total_loss / len(dataloader))

    return total_loss / len(dataloader)

def evaluate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['data'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, -1)
            total += labels.size(0) * labels.size(1)  # 总标签数量
            correct += (predicted == labels).sum().item()

            epoch_accuracy.append(correct / total)

    return correct / total

if __name__ == "__main__":

    data_files = ['ccq.npy', 'gyf.npy', 'pl.npy', 'tyt.npy', 'whr.npy', 'wl.npy', 'wm.npy']
    label_files = ['ccq_label.npy', 'gyf_label.npy', 'pl_label.npy', 'tyt_label.npy', 'whr_label.npy', 'wl_label.npy',
                   'wm_label.npy']

    data_files = [f'energy/{file}' for file in data_files]
    label_files = [f'wavelet/{file}' for file in label_files]

    batch_size = 16
    num_epochs = 3

    k_folds = 10
    kfold = KFold(n_splits=k_folds, shuffle=True)

    eeg_dataset = EEGDataset(data_files, label_files, window_size=32, step_size=1)

    results = {"fold": [], "epoch": [], "loss": [], "accuracy": []}

    for fold, (train_ids, test_ids) in enumerate(kfold.split(eeg_dataset)):

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        train_dataloader = DataLoader(eeg_dataset, batch_size=batch_size, sampler=train_subsampler)
        test_dataloader = DataLoader(eeg_dataset, batch_size=batch_size, sampler=test_subsampler)

        model = Waveformer().to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

        for epoch in range(num_epochs):
            loss = train(model, train_dataloader, loss_fn, optimizer, device)
            accuracy = evaluate_accuracy(model, test_dataloader, device)

            results["fold"].append(fold)
            results["epoch"].append(epoch)
            results["loss"].append(loss)
            results["accuracy"].append(accuracy)

    plt.figure(figsize=(14, 7))
    for fold in range(k_folds):
        # 筛选当前折的数据
        fold_results = {key: np.array(val)[np.array(results["fold"]) == fold] for key, val in results.items()}

        plt.subplot(1, 2, 1)
        plt.plot(fold_results["epoch"], fold_results["accuracy"], label=f'Fold {fold + 1}')
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(fold_results["epoch"], fold_results["loss"], label=f'Fold {fold + 1}')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

    plt.subplot(1, 2, 1)
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.legend()
    plt.grid(True)

    plt.savefig('CrossValidationResults.png')