import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt') 
video_path = r"C:\Users\lvvig\Downloads\train station video.mp4"
cap = cv2.VideoCapture(video_path)

line_y = 450 

# NEW: Set to store unique IDs of people who have landed on the platform
landed_ids = set()

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.resize(frame, (1020, 600))

    # --- THE CHANGE IS HERE ---
    # We use .track() instead of just calling model()
    # persist=True tells YOLO to remember IDs from the previous frame
    results = model.track(frame, persist=True, verbose=False)

    for r in results:
        # Check if any boxes were actually tracked (avoid errors if screen is empty)
        if r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            ids = r.boxes.id.cpu().numpy().astype(int)
            clss = r.boxes.cls.cpu().numpy().astype(int)

            for box, id, cls in zip(boxes, ids, clss):
                x1, y1, x2, y2 = box
                label = model.names[cls]
                
                # Center point calculation
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                if label == "person":
                    if cy > line_y:
                        color = (0, 255, 0)
                        status = f"ID:{id} ON PLATFORM"
                        # Add the ID to our set to count them
                        landed_ids.add(id)
                    else:
                        color = (0, 255, 255)
                        status = f"ID:{id} WAITING"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, status, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # DRAW THE LIVE COUNTER ON TOP
    count_text = f"Total Passengers Landed: {len(landed_ids)}"
    cv2.putText(frame, count_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    cv2.imshow('Smart Station Analytics', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()