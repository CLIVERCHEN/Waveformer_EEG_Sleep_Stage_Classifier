import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from models.CAE import ConvAutoencoder
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log_normalize(tensor):
    tensor_abs = torch.abs(tensor)
    return torch.log1p(tensor_abs)

def train_autoencoder(dataloader, epochs, criterion, optimizer):
    loss_history = []
    for epoch in range(epochs):
        for data, _ in dataloader:
            data = data.to(device)
            output = autoencoder(data)
            loss = criterion(output, data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_history.append(loss.item())
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    return loss_history

files = ['ccq.npy', 'gyf.npy', 'pl.npy', 'tyt.npy', 'whr.npy', 'wl.npy', 'wm.npy']

autoencoder = ConvAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

total_loss_history = []

for _ in range(10):
    tensor = []
    for file in random.sample(files, 3):
        data = np.load(f'wavelet16/{file}')
        for coe in data:
            tensor.append(coe)

    tensor = np.array(tensor)
    tensor = torch.Tensor(tensor)
    tensor = log_normalize(tensor).to(device)
    dataset = TensorDataset(tensor, tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    loss_history = train_autoencoder(dataloader, 30, criterion, optimizer)
    total_loss_history.extend(loss_history)

torch.save(autoencoder.state_dict(), 'autoencoder.pth')

plt.plot(total_loss_history)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('CAE_training.png')
plt.show()
