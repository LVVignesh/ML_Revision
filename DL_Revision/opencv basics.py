import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

# 1. Download the image from the URL
url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
resp = urllib.request.urlopen(url)
image_bytes = np.asarray(bytearray(resp.read()), dtype="uint8")

# 2. Decode the image bytes into an OpenCV image (BGR)
image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

# Now your original code will work!
print("Image Shape:", image.shape)
print("Data Type:", image.dtype)