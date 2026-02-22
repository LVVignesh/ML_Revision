import cv2
import os
import time

# 1. SETUP: Create a folder to store your "Training Data"
output_dir = "my_face_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. LOAD INTELLIGENCE
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
img_count = 0

print("Press 's' to save a face sample, or 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success: break

    # Pre-processing for the model
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw the visual feedback
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # --- THE DEEP LEARNING BRIDGE: ROI & RESIZING ---
        # 3. Crop the face (The ROI logic you learned!)
        face_roi = frame[y:y+h, x:x+w]
        
        # 4. Resize to a standard DL size (e.g., 224x224 for ResNet/ViT)
        # This is mandatory for Deep Learning tensors!
        face_resized = cv2.resize(face_roi, (224, 224))

        # 5. UI Feedback
        cv2.imshow('Face to be Saved', face_resized)

    cv2.imshow('Main Feed', frame)

    # Keyboard Logic
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and len(faces) > 0:
        # Save the processed image to your folder
        img_name = f"{output_dir}/face_{int(time.time())}.jpg"
        cv2.imwrite(img_name, face_resized)
        img_count += 1
        print(f"Saved image {img_count} to {img_name}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()