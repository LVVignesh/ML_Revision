# =========================================================
# RESNET18 PARTIAL FINE-TUNING ON CIFAR-10
# =========================================================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================================================
# DATA TRANSFORMS
# =========================================================

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=2
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=False, num_workers=2
)

# =========================================================
# LOAD PRETRAINED RESNET18
# =========================================================

model = torchvision.models.resnet18(weights="IMAGENET1K_V1")

# Replace final classifier
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)

model = model.to(device)

# =========================================================
# STEP 1: FREEZE ALL LAYERS
# =========================================================

for param in model.parameters():
    param.requires_grad = False

# Unfreeze only classifier head
for param in model.fc.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# =========================================================
# TRAIN CLASSIFIER HEAD FIRST
# =========================================================

epochs = 3
print("\n=== Training Classifier Head ===")

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(trainloader):.4f}")

# =========================================================
# STEP 2: PARTIAL UNFREEZE (layer4)
# =========================================================

print("\n=== Partial Unfreeze: layer4 ===")

for param in model.layer4.parameters():
    param.requires_grad = True

# Lower learning rate for fine-tuning
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

epochs = 5
loss_history = []

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(trainloader)
    loss_history.append(avg_loss)

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")


print("\n=== FULL UNFREEZE: Entire Backbone ===")

# 1. Unfreeze everything
for param in model.parameters():
    param.requires_grad = True

# 2. Set up the optimizer
optimizer = torch.optim.Adam([
    {"params": model.layer1.parameters(), "lr": 1e-5},
    {"params": model.layer2.parameters(), "lr": 1e-5},
    {"params": model.layer3.parameters(), "lr": 1e-5},
    {"params": model.layer4.parameters(), "lr": 5e-5},
    {"params": model.fc.parameters(), "lr": 1e-4}
])

# --- CRITICAL FIX: Define the list before using it ---
full_losses = [] 

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    # Using 'trainloader' as defined in your earlier code
    for images, labels in trainloader: 
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(trainloader)
    full_losses.append(epoch_loss) # This will work now!
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")

# =========================================================
# EVALUATION
# =========================================================

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\nFinal Test Accuracy: {accuracy:.2f}%")

# =========================================================
# LOSS PLOT
# =========================================================

plt.plot(loss_history)
plt.title("Fine-Tuning Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
