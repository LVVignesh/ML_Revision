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

# ==========================================
# Convert to Grayscale  
# ==========================================

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.imshow(image_gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.show()