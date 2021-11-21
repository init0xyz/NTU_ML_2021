import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.linear import Linear
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder
from torch.optim import lr_scheduler, optimizer
from tqdm.auto import tqdm
import torch.nn.functional as F

train_tfm = transforms.Compose([
    transforms.RandomRotation(40),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=0.2),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class Classifier_Alex_net(nn.Module):
    def __init__(self):
        super(Classifier_Alex_net, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64 ,3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(256, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(256, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512*5*5, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.flatten(1)
        x = self.fc_layers(x)
        return x


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 32
    train_set = DatasetFolder("~/data/food-11-big/training", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
    valid_set = DatasetFolder("~/data/food-11-big/validation", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
    test_set = DatasetFolder("~/data/food-11-big/testing", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers = 4, pin_memory = True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers = 4, pin_memory = True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = Classifier_Alex_net().to(device)
    model.device = device

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    n_epochs = 80
    best_acc = 0.0
    train_acc_list = []
    valid_acc_list = []

    for epoch in range(n_epochs):

        model.train()

        train_loss = []
        train_accs = []

        for batch in train_loader:
            imgs, labels = batch
            logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            train_loss.append(loss.item())
            train_accs.append(acc)
        
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        
        train_acc_list.append(float(train_acc))
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
        model.eval()
        valid_loss = []
        valid_accs = []

        for batch in (valid_loader):

            imgs, labels = batch
            with torch.no_grad():
                logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            valid_loss.append(loss.item())
            valid_accs.append(acc)

        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        valid_acc_list.append(valid_acc)
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
        if valid_acc > best_acc:
            torch.save(model.state_dict(), "base_model_alex.pt")
            best_acc = valid_acc





