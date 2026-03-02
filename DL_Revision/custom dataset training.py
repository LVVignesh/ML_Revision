# import os
# from ultralytics import YOLO

# # 1. SETUP FOLDERS (Automatically)
# base_path = "Phase3_Fast"
# os.makedirs(f"{base_path}/train/images", exist_ok=True)
# os.makedirs(f"{base_path}/train/labels", exist_ok=True)

# # 2. CREATE THE "MAP" (data.yaml)
# yaml_content = f"""
# path: {os.path.abspath(base_path)}
# train: train/images
# val: train/images
# nc: 1
# names: ['door']
# """
# with open(f"{base_path}/data.yaml", "w") as f:
#     f.write(yaml_content)

# # 3. CREATE "FAKE" LABELS (So you don't have to draw boxes)
# # This creates a label file for frame_0, frame_1, frame_2, frame_3
# for i in range(4):
#     with open(f"{base_path}/train/labels/frame_{i}.txt", "w") as f:
#         f.write("0 0.5 0.5 0.8 0.8") # Just a big box in the middle

# print("✅ Folders and Labels prepared. Now move your 4 images into Phase3_Fast/train/images!")
# print("Once moved, press Enter to start training...")
# input()

# # 4. START TRAINING
# model = YOLO('yolov8n.pt')
# model.train(data=f"{base_path}/data.yaml", epochs=3, imgsz=640)

# print("🎉 DONE! You finished Phase 3. Your 'best.pt' is in runs/detect/train/weights/")


import cv2
from ultralytics import YOLO
import os

# ==========================================
# 1. SETUP MODELS
# ==========================================
# Load the standard people detector
people_model = YOLO('yolov8n.pt') 

# Load YOUR custom train door detector
# Using the exact path from your training results
door_model_path = r'C:\Users\lvvig\runs\detect\train\weights\best.pt'

if not os.path.exists(door_model_path):
    print(f"ERROR: Could not find {door_model_path}")
    print("Please make sure you haven't moved the file, or copy it to your script folder.")
    exit()

door_model = YOLO(door_model_path)

# ==========================================
# 2. VIDEO SETUP
# ==========================================
video_path = r"C:\Users\lvvig\Downloads\train station video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video file. Check the path!")
    exit()

print("Starting AI Smart Station... Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # ==========================================
    # 3. RUN AI DETECTION
    # ==========================================
    # Detect people (Standard YOLO)
    person_results = people_model(frame, verbose=False)
    
    # Detect doors (Your trained YOLO)
    # We use a low confidence (0.1) because we trained on a small dataset
    door_results = door_model(frame, verbose=False, conf=0.1)

    person_count = 0
    door_open = False

    # Process People Results
    for r in person_results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            if people_model.names[class_id] == 'person':
                person_count += 1

    # Process Door Results
    for r in door_results:
        if len(r.boxes) > 0:
            door_open = True

    # ==========================================
    # 4. SAFETY LOGIC ENGINE
    # ==========================================
    if door_open and person_count > 0:
        status = "BOARDING: DOORS OPEN"
        color = (0, 255, 0) # Green for Safe Boarding
    elif not door_open and person_count > 0:
        status = "DANGER: PERSON ON PLATFORM"
        color = (0, 0, 255) # Red for Danger
    else:
        status = "STATION CLEAR"
        color = (255, 255, 255) # White for Idle

    # ==========================================
    # 5. DRAW THE DASHBOARD
    # ==========================================
    # Draw a black semi-transparent header
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (700, 150), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Put Text on Screen
    cv2.putText(frame, f"STATUS: {status}", (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    
    cv2.putText(frame, f"PASSENGER COUNT: {person_count}", (20, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the final result
    cv2.imshow("Vignesh's AI Smart Station", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("System Shutdown Successfully.")