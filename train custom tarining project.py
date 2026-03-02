import cv2
from ultralytics import YOLO

# 1. Load the model
model = YOLO('yolov8n.pt') 

video_path = r"C:\Users\lvvig\Downloads\train station video.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 2. Run detection
    results = model(frame, verbose=False)
    
    person_count = 0
    door_detected = False # We will pretend 'suitcases' or 'bench' are doors for testing
    
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            
            if label == 'person':
                person_count += 1
            
            # MOCK LOGIC: Let's pretend a 'handbag' or 'suitcase' is a door
            if label in ['handbag', 'suitcase', 'backpack']:
                door_detected = True

    # 3. SAFETY LOGIC ENGINE
    status = "WAITING"
    color = (255, 255, 255) # White

    if door_detected and person_count > 0:
        status = "BOARDING IN PROGRESS"
        color = (0, 255, 0) # Green
    elif not door_detected and person_count > 2:
        status = "DANGER: PEOPLE ON TRACKS?"
        color = (0, 0, 255) # Red

    # 4. Draw the Dashboard
    cv2.rectangle(frame, (20, 20), (450, 120), (0, 0, 0), -1) # Black background for text
    cv2.putText(frame, f"Status: {status}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Passengers: {person_count}", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Smart Station Safety System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()