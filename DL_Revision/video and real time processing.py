import cv2

# 1. Connect to the Webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    success, frame = cap.read()
    if not success:
        break

    # --- ALL YOUR CV LOGIC GOES HERE ---
    # Example: Convert each frame to Gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Example: Detect Edges in real-time
    edges = cv2.Canny(gray, 100, 200)
    # ------------------------------------

    # Display the resulting frame
    cv2.imshow('Real-time Edge Detection', edges)

    # Press 'q' on the keyboard to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()