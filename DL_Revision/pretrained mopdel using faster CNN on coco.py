# ==========================================
# INSTALL (Colab only)
# ==========================================

#pip install pycocotools

# ==========================================
# IMPORTS
# ==========================================

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import requests
from io import BytesIO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==========================================
# LOAD PRETRAINED DETECTION MODEL
# ==========================================

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

# ==========================================
# LOAD IMAGE FROM INTERNET
# ==========================================

url = "https://images.unsplash.com/photo-1503023345310-bd7c1de61c7d"
response = requests.get(url)
image = Image.open(BytesIO(response.content)).convert("RGB")

transform = transforms.Compose([
    transforms.ToTensor()
])

img_tensor = transform(image).to(device)

# ==========================================
# RUN INFERENCE
# ==========================================

with torch.no_grad():
    prediction = model([img_tensor])

# ==========================================
# VISUALIZE PREDICTIONS
# ==========================================

fig, ax = plt.subplots(1, figsize=(10,8))
ax.imshow(image)

boxes = prediction[0]['boxes'].cpu()
scores = prediction[0]['scores'].cpu()
labels = prediction[0]['labels'].cpu()

for i in range(len(boxes)):
    if scores[i] > 0.5:  # confidence threshold
        box = boxes[i]
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2]-box[0],
            box[3]-box[1],
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)

plt.title("Predicted Bounding Boxes")
plt.show()
