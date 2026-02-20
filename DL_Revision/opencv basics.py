# ==========================================
# OpenCV Basics (Fixed Version)
# ==========================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

# ==========================================
# Download Image First
# ==========================================

url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
urllib.request.urlretrieve(url, "lena.jpg")

# ==========================================
# Load Image
# ==========================================

image = cv2.imread("lena.jpg")

print("Image Shape:", image.shape)
print("Data Type:", image.dtype)

# ==========================================
# Convert BGR → RGB
# ==========================================

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()

# 1. Create a copy so we don't ruin the original
annotated_img = image_rgb.copy()

# 2. Draw a Rectangle (Start Point, End Point, Color, Thickness)
# Color is RGB because we converted it earlier!
cv2.rectangle(annotated_img, (200, 200), (400, 400), (0, 255, 0), 5)

# 3. Draw a Circle (Center, Radius, Color, Thickness)
# Thickness = -1 fills the shape
cv2.circle(annotated_img, (255, 255), 50, (255, 0, 0), -1)

# 4. Put Text (Image, Text, Bottom-Left Corner, Font, Scale, Color, Thickness)
cv2.putText(annotated_img, "Target Acquired", (50, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

plt.imshow(annotated_img)
plt.show()

# 1. Define the coordinates for the face
# On the 512x512 Lena image, the face is roughly here:
y_start, y_end = 200, 400
x_start, x_end = 200, 400

# 2. Crop the image (Remember: Y/Rows come first!)
face_crop = image_rgb[y_start:y_end, x_start:x_end]

# 3. Show the result
plt.imshow(face_crop)
plt.title("Cropped ROI (Face)")
plt.show()