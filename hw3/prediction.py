import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder
from torch.optim import lr_scheduler
from tqdm.auto import tqdm
import torch.nn.functional as F

test_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

batch_size = 32
test_set = DatasetFolder("~/data/food-11-big/testing", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

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

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Classifier_Alex_net().to(device)
model.load_state_dict(torch.load("base_model_alex.pt"))
model.eval()
predictions = []

for batch in tqdm(test_loader):
    imgs, label = batch
    with torch.no_grad():
        logits = model(imgs.to(device))

    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

with open("predict.csv", 'w') as f:
    f.write('Id,Category\n')
    for i, y in  enumerate(predictions):
        f.write('{},{}\n'.format(i, y))