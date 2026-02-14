# =========================================================
# TRANSFER LEARNING WITH RESNET18 ON CIFAR-10
# =========================================================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# =========================================================
# STEP 1: DEVICE SETUP
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================================================
# STEP 2: DATA TRANSFORMS
# =========================================================
# ResNet expects 224x224 input
# CIFAR is 32x32 → so we resize

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

# =========================================================
# STEP 3: LOAD DATASET
# =========================================================

trainset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform_train
)

testset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform_test
)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# =========================================================
# STEP 4: LOAD PRETRAINED RESNET18
# =========================================================

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze all layers (feature extractor)
for param in model.parameters():
    param.requires_grad = False

# Replace final fully connected layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)

model = model.to(device)

# =========================================================
# STEP 5: LOSS & OPTIMIZER
# =========================================================

criterion = nn.CrossEntropyLoss()

# Only train final layer
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# =========================================================
# STEP 6: TRAINING LOOP
# =========================================================

epochs = 5
train_losses = []

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for images, labels in tqdm(trainloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(trainloader)
    train_losses.append(epoch_loss)

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f}")

# =========================================================
# STEP 7: EVALUATION
# =========================================================

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\nTest Accuracy: {accuracy:.2f}%")

# =========================================================
# STEP 8: PLOT TRAINING LOSS
# =========================================================

plt.plot(train_losses)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
