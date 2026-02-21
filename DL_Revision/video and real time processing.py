import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success: break

    # 1. Processing with extra smoothing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0) # Increased blur for stability
    edges = cv2.Canny(blurred, 30, 100) # Lower thresholds to catch more of you

    # 2. THE STABILIZER: Dilation & Erosion (Morphology)
    # This "heals" broken edge lines so the contour stays solid
    kernel = np.ones((5,5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    # 3. Contour Detection
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Using a slightly lower threshold for smaller objects like glasses
        if area > 600: 
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "STABLE OBJ", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display Stack
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    stacked = np.hstack((frame, edges_bgr))
    cv2.imshow('Stable Real-time Tracking', stacked)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()