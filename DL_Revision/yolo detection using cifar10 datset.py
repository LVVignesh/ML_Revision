# ==========================================
# SIMPLIFIED YOLO DETECTOR - CIFAR10
# ==========================================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==========================================
# DATA
# ==========================================

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

# ==========================================
# CREATE FAKE BOUNDING BOX TARGETS
# ==========================================

# Since CIFAR has no boxes,
# we pretend object occupies entire image.

def create_bbox_targets(batch_size):
    # center_x, center_y, width, height
    # normalized between 0 and 1
    return torch.tensor([[0.5, 0.5, 1.0, 1.0]] * batch_size).float().to(device)

# ==========================================
# YOLO-LIKE MODEL
# ==========================================

class YOLO_Simple(nn.Module):
    def __init__(self):
        super().__init__()

        # Backbone CNN
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )

        # Detection Head
        self.class_head = nn.Linear(128, 10)   # 10 classes
        self.box_head = nn.Linear(128, 4)      # bbox regression

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)

        class_logits = self.class_head(x)
        bbox = torch.sigmoid(self.box_head(x))  # normalized 0–1

        return class_logits, bbox


model = YOLO_Simple().to(device)

# ==========================================
# LOSS FUNCTIONS
# ==========================================

classification_loss = nn.CrossEntropyLoss()
bbox_loss = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# TRAINING
# ==========================================

num_epochs = 10
loss_history = []

for epoch in range(num_epochs):

    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        bbox_targets = create_bbox_targets(images.size(0))

        optimizer.zero_grad()

        class_logits, bbox_preds = model(images)

        loss_class = classification_loss(class_logits, labels)
        loss_box = bbox_loss(bbox_preds, bbox_targets)

        total_loss = loss_class + loss_box

        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

    epoch_loss = running_loss / len(train_loader)
    loss_history.append(epoch_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")

# ==========================================
# EVALUATION
# ==========================================

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        class_logits, _ = model(images)
        _, predicted = torch.max(class_logits, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\nTest Classification Accuracy: {accuracy:.2f}%")

# ==========================================
# PLOT LOSS
# ==========================================

plt.plot(loss_history)
plt.title("YOLO Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
