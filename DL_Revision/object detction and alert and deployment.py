import cv2
from ultralytics import YOLO
import os
import time

# --- PHASE 6: DEPLOYMENT PREP ---
# Create a folder to store safety reports automatically
if not os.path.exists('safety_logs'):
    os.makedirs('safety_logs')

# 1. SETUP MODELS
people_model = YOLO('yolov8n.pt') 
# Update this path if you moved your 'best.pt' file
door_model_path = r'C:\Users\lvvig\runs\detect\strong_door_model\weights\best.pt'
door_model = YOLO(door_model_path)

video_path = r"C:\Users\lvvig\Downloads\train station video.mp4"
cap = cv2.VideoCapture(video_path)

# Variables for Phase 5 Tracking & Phase 6 Logging
last_log_time = 0 

while cap.isOpened():
    success, frame = cap.read()
    if not success: 
        break

    # 2. RUN AI DETECTION
    # Phase 5: Multi-Object Tracking (keeps IDs consistent)
    person_results = people_model.track(frame, persist=True, verbose=False)
    # Using low confidence (0.1) to ensure the door is caught
    door_results = door_model(frame, verbose=False, conf=0.1)

    person_count = 0
    door_open = False

    # DRAW PEOPLE BOXES & TRACK IDs
    for r in person_results:
        if r.boxes.id is not None:
            ids = r.boxes.id.int().tolist()
            for box, id in zip(r.boxes, ids):
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue for people
                cv2.putText(frame, f"ID:{id}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # DRAW DOOR BOXES
    for r in door_results:
        if len(r.boxes) > 0:
            door_open = True
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green for doors

    # 3. ADVANCED SAFETY LOGIC
    if door_open:
        status, color = "BOARDING: SAFE", (0, 255, 0) # Green
    elif not door_open and person_count > 0:
        status, color = "DANGER: UNAUTHORIZED ACCESS", (0, 0, 255) # Red
        
        # PHASE 6: AUTOMATED INCIDENT REPORTING
        current_time = time.time()
        if current_time - last_log_time > 5: # Limit logging to once every 5 seconds
            log_path = f"safety_logs/violation_{int(current_time)}.jpg"
            cv2.imwrite(log_path, frame)
            last_log_time = current_time
            print(f"Incident Logged: {log_path}")
    else:
        status, color = "STATION CLEAR", (255, 255, 255) # White

    # 4. UPDATED DASHBOARD UI (The final layer)
    # Background rectangle
    cv2.rectangle(frame, (0, 0), (800, 110), (0, 0, 0), -1) 

    # Status text (changes color based on logic)
    cv2.putText(frame, f"LOGIC: {status}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    # Passenger count text
    cv2.putText(frame, f"PASSENGERS: {person_count}", (20, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 5. DISPLAY
    cv2.imshow("Vignesh's AI Smart Station (Phase 6)", frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()