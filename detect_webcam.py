import cv2
from ultralytics import YOLO

# Load YOLOv8 nano model (you can try 'yolov8s.pt' later)
model = YOLO("yolov8n.pt")

# Start the webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

while True:
    success, frame = cap.read()
    if not success:
        break

    # Run object detection
    results = model(frame)

    # Draw boxes and labels
    annotated_frame = results[0].plot()

    # Show the output
    cv2.imshow("YOLOv8 Webcam Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
