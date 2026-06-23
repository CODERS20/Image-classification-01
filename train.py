import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets

from torch.utils.data import Dataset, DataLoader, random_split

from model import ResNet, block

import os

if os.path.exists('/content/drive/MyDrive/ml-data'):
    data_directory = '/content/drive/MyDrive/ml-data' 
else:
    data_directory = './dataset'

#defining the image transformation

import torchvision.transforms.functional as TF

class increase_contrast(object):
    def __call__(self, img):
        return TF.adjust_contrast(img, contrast_factor=1.8)

transform = transforms.Compose([
    transforms.Resize((256,256)),
    increase_contrast(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=180),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root = data_directory,transform = transform)

#train test validation

total = len(dataset)
train_size = int(0.8 * total)
val_size = int(0.1 * total)
test_size = total - train_size-val_size

train_set,val_set,test_set = random_split(
    dataset,
    [train_size,val_size,test_size],
    generator=torch.Generator().manual_seed(40)
)

train_loader = DataLoader(train_set,batch_size=32,shuffle=True,num_workers=2,pin_memory=True)
val_loader = DataLoader(val_set,batch_size=32,shuffle=False)
test_loader = DataLoader(test_set,batch_size=32,shuffle=False)

transform = transforms.Compose([
    transforms.Resize((256,256)),
    increase_contrast(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=180),
    transforms.ToTensor()
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#creating the model

def ResNet50(img_channels=3, num_classes = 5):
  return ResNet(block,[3,4,6,3],img_channels,num_classes)

model = ResNet50(img_channels=3, num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)


#training
for epoch in range(30):
    print(f'Training epoch {epoch}..')

    model.train()

    running_loss = 0.0

    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Loss: {running_loss/len(train_loader):.4f}')