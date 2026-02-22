import cv2
from ultralytics import YOLO

# 1. Load a pre-trained Deep Learning model (YOLOv8 Nano - very fast)
model = YOLO('yolov8n.pt') 

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success: break

    # 2. DEEP LEARNING INFERENCE
    # This single line replaces all your Canny, Blur, and Cascade math!
    results = model(frame, stream=True)

    # 3. PARSING THE RESULTS
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract coordinates (The x, y, w, h you know so well now!)
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get class name (Person, Glasses, Chair, etc.)
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = round(float(box.conf[0]), 2)

            # 4. DRAWING (Back to your OpenCV skills)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(frame, f'{label} {conf}', (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    cv2.imshow('Phase 7: Deep Learning YOLO Inference', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()