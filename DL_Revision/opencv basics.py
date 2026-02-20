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

# 1. Convert to Grayscale
gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

# 2. Apply Thresholding
# This says: "If a pixel is brighter than 127, make it 255 (white). 
# Otherwise, make it 0 (black)."
ret, thresh_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Display both
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1); plt.imshow(gray_image, cmap='gray'); plt.title("Grayscale")
plt.subplot(1, 2, 2); plt.imshow(thresh_image, cmap='gray'); plt.title("Binary Threshold")
plt.show()

# 1. Blur the image to reduce noise (Kernel size must be odd, like 5x5)
blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

# 2. Canny Edge Detection (Image, MinThreshold, MaxThreshold)
edges = cv2.Canny(blurred, 50, 150)

# Display
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1); plt.imshow(blurred, cmap='gray'); plt.title("Blurred (Smooth)")
plt.subplot(1, 2, 2); plt.imshow(edges, cmap='gray'); plt.title("Canny Edges")
plt.show()


#contours 

# 1. Find the contours (as we did before)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 2. Create the clean canvas
contour_img = image_rgb.copy()

# 3. INSERT THE FILTERING LOGIC HERE
for cnt in contours:
    area = cv2.contourArea(cnt)
    
    # Only process shapes that are big enough to be "real" objects
    if area > 100: 
        cv2.drawContours(contour_img, [cnt], -1, (0, 255, 0), 2)
        
        # PRO TIP: Let's also print the area of the objects found
        print(f"Object found with area: {area}")

# 4. Show the result
plt.imshow(contour_img)
plt.title("Filtered Contours (>100px)")
plt.show()