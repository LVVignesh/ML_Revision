import cv2
from ultralytics import YOLO
import os

# 1. SETUP MODELS
people_model = YOLO('yolov8n.pt') 

# UPDATE: Pointing to the new "Strong" brain from your 50-epoch training
door_model_path = r'C:\Users\lvvig\runs\detect\strong_door_model\weights\best.pt'

if not os.path.exists(door_model_path):
    print(f"Error: Could not find {door_model_path}")
    print("Wait for the boost_train.py script to finish first!")
    exit()

door_model = YOLO(door_model_path)

video_path = r"C:\Users\lvvig\Downloads\train station video.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 2. RUN AI DETECTION
    person_results = people_model(frame, verbose=False)
    
    # UPDATE: Using a slightly more sensitive confidence (0.05)
    door_results = door_model(frame, verbose=False, conf=0.05)

    person_count = 0
    door_open = False

    # --- DRAW BOXES FOR PEOPLE ---
    for r in person_results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            if people_model.names[class_id] == 'person':
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue box
                cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # --- DRAW BOXES FOR DOORS ---
    for r in door_results:
        if len(r.boxes) > 0:
            door_open = True
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green box
                cv2.putText(frame, "Train Door", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 3. SAFETY LOGIC
    if door_open and person_count > 0:
        status = "BOARDING: DOORS OPEN"
        color = (0, 255, 0)
    elif not door_open and person_count > 0:
        status = "DANGER: PERSON ON PLATFORM"
        color = (0, 0, 255)
    else:
        status = "STATION CLEAR"
        color = (255, 255, 255)

    # 4. DRAW DASHBOARD
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (700, 150), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    cv2.putText(frame, f"STATUS: {status}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    cv2.putText(frame, f"PASSENGER COUNT: {person_count}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Vignesh's AI Smart Station", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()