import cv2
from ultralytics import YOLO

# 1. Load YOLOv8 (Intelligence)
model = YOLO('yolov8n.pt') 

# 2. Use your specific file path (Note the 'r' for raw string)
video_path = r"C:\Users\lvvig\Downloads\train station video.mp4"
cap = cv2.VideoCapture(video_path)

# Define a "Safety Line" (Adjust 450 based on your video's perspective)
line_y = 450 

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # Resize so it fits on your monitor comfortably
    frame = cv2.resize(frame, (1020, 600))

    # 3. Detect objects
    results = model(frame, verbose=False)

    # Draw the Platform Safety Line
    cv2.line(frame, (0, line_y), (1020, line_y), (0, 0, 255), 3)
    cv2.putText(frame, "PLATFORM EDGE", (10, line_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    for r in results:
        for box in r.boxes:
            # Get coords and label
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            conf = float(box.conf[0])

            # Calculate Center Point (The "Feet" of the passenger)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if label == "person" and conf > 0.4:
                # Logic: If person's center is below the red line, they are "Safe on Platform"
                if cy > line_y:
                    color = (0, 255, 0) # Green for Safe
                    status = "ON PLATFORM"
                else:
                    color = (0, 255, 255) # Yellow for Waiting/Inside Train
                    status = "WAITING"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 5, color, -1)
                cv2.putText(frame, status, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Smart Station Analytics', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()