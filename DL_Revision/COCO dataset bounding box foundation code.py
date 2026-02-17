import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# ==========================================
# DEVICE
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==========================================
# LOAD DATASET
# ==========================================

transform = transforms.ToTensor()

dataset = torchvision.datasets.CocoDetection(
    root="./coco/val2017",
    annFile="./coco/annotations/instances_val2017.json",
    transform=transform
)

print("Total images:", len(dataset))


# ==========================================
# VISUALIZE ONE IMAGE WITH BOXES
# ==========================================

image, targets = dataset[10]

# Convert tensor to PIL image
img = image.permute(1, 2, 0).numpy()

fig, ax = plt.subplots(1)
ax.imshow(img)

for obj in targets:
    bbox = obj["bbox"]  # [x, y, width, height]
    rect = patches.Rectangle(
        (bbox[0], bbox[1]),
        bbox[2],
        bbox[3],
        linewidth=2,
        edgecolor='red',
        facecolor='none'
    )
    ax.add_patch(rect)

plt.title("COCO Detection Example")
plt.show()


