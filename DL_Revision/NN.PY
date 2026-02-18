# ==========================================
# Fully Updated YOLO-style Detector (Fixed)
# ==========================================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==========================================
# 1. DOWNLOAD DATASET
# ==========================================
if not os.path.exists("PennFudanPed"):
   # You can download the PennFudanPed dataset from:
   # https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip
   # Unzip it and place the "PennFudanPed" folder in the same directory as this script.
   print("Please download the PennFudanPed dataset and place it in the same directory as this script.")

# ==========================================
# 2. DATASET (With Coordinate Scaling)
# ==========================================
class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = sorted(os.listdir(os.path.join(root, "PNGImages")))
        self.masks = sorted(os.listdir(os.path.join(root, "PedMasks")))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size # Original size
        
        mask = np.array(Image.open(mask_path))
        obj_ids = np.unique(mask)[1:] # Remove background

        boxes = []
        for obj_id in obj_ids:
            pos = np.where(mask == obj_id)
            # Scale coordinates to match the 224x224 resize
            xmin = np.min(pos[1]) * (224.0 / orig_w)
            xmax = np.max(pos[1]) * (224.0 / orig_w)
            ymin = np.min(pos[0]) * (224.0 / orig_h)
            ymax = np.max(pos[0]) * (224.0 / orig_h)
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        if self.transforms:
            img = self.transforms(img)

        return img, boxes

    def __len__(self):
        return len(self.imgs)

# Standardize all images to 224x224
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images, 0), targets

dataset = PennFudanDataset("PennFudanPed", transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# ==========================================
# 3. MODEL (Fixed Linear Layer Math)
# ==========================================
class SimpleYOLO(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 112
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 56
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 28
        )
        # After 3 MaxPools, 224 becomes 28. Input to Linear = 64 * 28 * 28
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 4) # Output: [xmin, ymin, xmax, ymax]
        )

    def forward(self, x):
        return self.head(self.backbone(x))

model = SimpleYOLO().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# ==========================================
# 4. TRAINING
# ==========================================
print("Starting Training...")
for epoch in range(30): # Increased to 10 for better results
    model.train()
    total_loss = 0
    for imgs, targets in loader:
        imgs = imgs.to(device)
        
        # Get first person box for each image in batch
        gt_boxes = torch.stack([t[0] if len(t) > 0 else torch.zeros(4) for t in targets]).to(device)
        
        preds = model(imgs)
        loss = criterion(preds, gt_boxes)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/10] Loss: {total_loss/len(loader):.4f}")

# ==========================================
# 5. INFERENCE & VISUALIZATION
# ==========================================
model.eval()
img, _ = dataset[5] # Take a sample
with torch.no_grad():
    pred = model(img.unsqueeze(0).to(device)).cpu().numpy()[0]

plt.imshow(img.permute(1, 2, 0))
plt.gca().add_patch(plt.Rectangle((pred[0], pred[1]), pred[2]-pred[0], pred[3]-pred[1], 
                                  fill=False, color='red', linewidth=3))
plt.title("YOLO Detection Result")
plt.show()