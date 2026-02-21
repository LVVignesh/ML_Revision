import cv2
import numpy as np

# 1. Start Webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # --- PHASE 2: PROCESSING ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # --- PHASE 3: CONTOURS ---
    # We find contours on the EDGES, but we draw them on the COLOR frame
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000: # Increased to 1000 to keep the screen cleaner
            # Draw on the 'frame' variable
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            print(f"Total contours found: {len(contours)}") # Add this!
    for cnt in contours:
   
    # --- THE PROFESSIONAL STACK ---
    # Since 'edges' is 1-channel (gray) and 'frame' is 3-channel (color),
    # we must convert edges to BGR just so we can stack them together.
      edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
      stacked_output = np.hstack((frame, edges_bgr))

    # 5. Display the stacked result
    cv2.imshow('Left: Final Result | Right: Edge Detection (The Brain)', stacked_output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()