import cv2
from ultralytics import YOLO

# 1. Load the "Brain"
model = YOLO('yolov8n.pt') 

# 2. Setup Video Source
video_path = r"C:\Users\lvvig\Downloads\train station video.mp4"
cap = cv2.VideoCapture(video_path)

# 3. Setup Video EXPORTER (The "Recorder")
# This creates a file named 'Station_Analysis_Output.avi' in your current folder
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('Station_Analysis_Output.avi', fourcc, 20.0, (1020, 600))

# Logic Variables
line_y = 450 
landed_ids = set() # Our notebook to remember unique people

print("Processing video... Press 'q' to stop and save.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: 
        break

    # Resize for consistent processing
    frame = cv2.resize(frame, (1020, 600))

    # 4. Run Tracking (Memory-enabled detection)
    results = model.track(frame, persist=True, verbose=False)

    # Draw the boundary line for visual reference
    cv2.line(frame, (0, line_y), (1020, line_y), (0, 0, 255), 3)
    cv2.putText(frame, "PLATFORM EDGE", (10, line_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    for r in results:
        # Only proceed if YOLO actually found/tracked something
        if r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            ids = r.boxes.id.cpu().numpy().astype(int)
            clss = r.boxes.cls.cpu().numpy().astype(int)

            for box, id, cls in zip(boxes, ids, clss):
                x1, y1, x2, y2 = box
                label = model.names[cls]
                
                # Center point (the "Feet" of the person)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                if label == "person":
                    # Check if they are below the line (Landed)
                    if cy > line_y:
                        color = (0, 255, 0) # Green
                        status = f"ID:{id} ON PLATFORM"
                        landed_ids.add(id) # Add to unique count
                    else:
                        color = (0, 255, 255) # Yellow
                        status = f"ID:{id} WAITING"

                    # Draw the visuals
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(frame, (cx, cy), 5, color, -1)
                    cv2.putText(frame, status, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 5. Display & Record the Counter
    count_text = f"Total Passengers Landed: {len(landed_ids)}"
    cv2.putText(frame, count_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    # Show the frame live
    cv2.imshow('Smart Station Analytics', frame)
    
    # SAVE the frame to the output video file
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 6. Cleanup
print(f"Finished! Total unique passengers detected: {len(landed_ids)}")
cap.release()
out.release() # This "closes" the video file so you can watch it
cv2.destroyAllWindows()