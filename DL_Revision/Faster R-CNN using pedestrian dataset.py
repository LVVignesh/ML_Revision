# ==========================================
# PennFudan Object Detection - Faster R-CNN
# ==========================================

import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==========================================
# Download Dataset
# ==========================================
# You can download the PennFudanPed dataset from:
# https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip
# Unzip it and place the "PennFudanPed" folder in the same directory as this script.

# ==========================================
# Dataset Class
# ==========================================

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(mask_path))

        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]

        boxes = []
        for i in range(len(obj_ids)):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(obj_ids),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

# ==========================================
# DataLoader
# ==========================================

def collate_fn(batch):
    return tuple(zip(*batch))

transform = T.Compose([T.ToTensor()])

dataset = PennFudanDataset("PennFudanPed", transform)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn
)

# ==========================================
# Model
# ==========================================

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features,
    num_classes
)

model.to(device)

# ==========================================
# Optimizer
# ==========================================

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# ==========================================
# Training
# ==========================================

num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {total_loss:.4f}")

# ==========================================
# Inference
# ==========================================

model.eval()
img, _ = dataset[0]

with torch.no_grad():
    prediction = model([img.to(device)])

img_np = img.permute(1,2,0).numpy()

plt.figure(figsize=(6,6))
plt.imshow(img_np)

for box in prediction[0]['boxes']:
    xmin, ymin, xmax, ymax = box.cpu()
    plt.gca().add_patch(
        plt.Rectangle((xmin, ymin),
                      xmax - xmin,
                      ymax - ymin,
                      fill=False,
                      color='red',
                      linewidth=2)
    )

plt.title("Detected Pedestrians")
plt.show()
