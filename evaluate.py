import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets

from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from model import ResNet, block

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def ResNet50(img_channels=3, num_classes = 5):
  return ResNet(block,[3,4,6,3],img_channels,num_classes)
model = ResNet50(img_channels=3, num_classes=5).to(device)

correct = 0
total = 0

model.eval()

with torch.no_grad():
  for data in test_loader:
    images,labels = data

    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)
    _,predicted = torch.max(outputs,1)
    total += labels.size(0)
    correct += (predicted ==labels).sum().item()
accuracy = 100*correct/total
print(f'Accuracy score:{accuracy}')



all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=dataset.classes,
            yticklabels=dataset.classes)
plt.savefig('/content/confusion_matrix.png', dpi=300, bbox_inches='tight')

plt.xlabel('Predicted Objects')
plt.ylabel('True Objects')
plt.title('Confusion Matrix')
plt.show()