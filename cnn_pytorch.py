# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
#
#
# import torch
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from torch import nn, optim
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, random_split
# from torchvision import transforms
# from tqdm import tqdm
#
#
#
# print(torch.cuda.is_available())
# device = torch.device('cpu')
#
# def get_dataloaders(batch_size):
#     df = pd.read_csv('data/train.csv')
#     test = pd.read_csv('data/test.csv')
#     sub = pd.read_csv('data/sample_submission.csv')
#
#     x_train = df.iloc[:, 1:].values.reshape(-1, 28, 28, 1).astype(np.float32)
#     y_train = df.iloc[:, 0].values
#
#     scaler = MinMaxScaler()
#     x_train = x_train.reshape(-1, 28 * 28)
#     x_train = scaler.fit_transform(x_train).reshape(-1, 28, 28, 1)
#
#     test_x = test.values.reshape(-1, 28 * 28)
#     test_x = scaler.transform(test_x).reshape(-1, 28, 28, 1)
#
#     class CustomDataset(Dataset):
#         def __init__(self, images, labels=None, transform=None):
#             self.images = images
#             self.labels = labels
#             self.transform = transform
#
#         def __len__(self):
#             return len(self.images)
#
#         def __getitem__(self, idx):
#             image = self.images[idx]
#             label = self.labels[idx] if self.labels is not None else -1
#             if self.transform:
#                 image = self.transform(image)
#             return image, label
#
#     transform = transforms.Compose([
#         transforms.ToTensor()
#     ])
#
#     dataset = CustomDataset(x_train, y_train, transform=transform)
#     train_size = int(0.8 * len(dataset))
#     val_size = len(dataset) - train_size
#     train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
#
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#
#     return train_loader, val_loader
#
#
# def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
#     train_losses = []
#     val_losses = []
#     valid_accuracies = []
#     for epoch in range(epochs):
#         model.train()
#         loss_sum = 0
#         for xb, yb in tqdm(train_dl):
#             xb, yb = xb.to(device), yb.to(device)
#             loss = loss_func(model(xb), yb)
#             loss_sum += loss.item()
#
#             loss.backward()
#             opt.step()
#             opt.zero_grad()
#         train_losses.append(loss_sum / len(train_dl))
#
#         model.eval()
#         loss_sum = 0
#         correct = 0
#         num = 0
#         with torch.no_grad():
#             for xb, yb in valid_dl:
#                 xb, yb = xb.to(device), yb.to(device)
#                 probs = model(xb)
#                 loss_sum += loss_func(probs, yb).item()
#
#                 _, preds = torch.max(probs, axis=-1)
#                 correct += (preds == yb).sum().item()
#                 num += len(xb)
#
#         val_losses.append(loss_sum / len(valid_dl))
#         valid_accuracies.append(correct / num)
#
#     return train_losses, val_losses, valid_accuracies
#
#
# def plot_trainig(train_losses, valid_losses, valid_accuracies):
#     plt.figure(figsize=(12, 9))
#     plt.subplot(2, 1, 1)
#     plt.xlabel('epoch')
#     plt.plot(train_losses, label='train_loss')
#     plt.plot(valid_losses, label='valid_loss')
#     plt.legend()
#
#     plt.subplot(2, 1, 2)
#     plt.xlabel('epoch')
#     plt.plot(valid_accuracies, label='valid accuracy')
#     plt.legend()
#     plt.show()
#
#
# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(64)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.dropout = nn.Dropout(0.3)
#         self.fc1 = nn.Linear(64 * 3 * 3, 200)
#         self.fc2 = nn.Linear(200, 10)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#         x = self.pool(F.relu(self.bn3(self.conv3(x))))
#         x = x.view(-1, 64 * 3 * 3)
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x
#
#
# model = Model().to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
#
# info = fit(10, model, criterion, optimizer, *get_dataloaders(32))
# plot_trainig(*info)

import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, random_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

df = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
sub = pd.read_csv('data/sample_submission.csv')

x_train = df.iloc[:, 1:].values.reshape(-1, 28, 28, 1)
y_train = df.iloc[:, 0].values

scaler = MinMaxScaler()
x_train = x_train.reshape(-1, 28 * 28)
x_train = scaler.fit_transform(x_train).reshape(-1, 28, 28, 1).astype(np.float32)

test_x = test.values.reshape(-1, 28 * 28)
test_x = scaler.transform(test_x).reshape(-1, 28, 28, 1).astype(np.float32)


class CustomDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx] if self.labels is not None else -1
        if self.transform:
            image = self.transform(image)
        return image, label


transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = CustomDataset(x_train, y_train, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout(0.1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout3 = nn.Dropout(0.1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 14 * 14, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))
        out = F.relu((self.conv2(out)))
        out = F.relu((self.conv3(out)))
        out = out + identity
        out = self.pool(out)
        out = out.view(-1, 64 * 14 * 14)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


model = CNNModel().to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
patience = 5
best_val_loss = float('inf')
counter = 0

train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100 * correct / total

    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)

    print(f"Epoch {epoch + 1}/{epochs}, "
          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
          f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break

plt.figure(figsize=(13, 5))
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Val Loss')
plt.title("Model Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(13, 5))
plt.plot(train_acc_history, label='Train Accuracy')
plt.plot(val_acc_history, label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

test_dataset = CustomDataset(test_x, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.load_state_dict(torch.load('best_model.pth'))
model.eval()

pred_labels = []
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        pred_labels.extend(predicted.cpu().numpy())

sub['Label'] = pred_labels
sub.to_csv("data/sample_submission.csv", index=False)
print(sub.head())

image = (np.array(
    Image.open('C:\\Users\\pasag\\Desktop\\digit photo\\7.jpg').convert('L').resize((28, 28))) / 255.0).astype(
    np.float32)
image = torch.tensor(image).unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad():
    model.eval()
    output = model(image)
    print(torch.argmax(output, dim=1).item())
