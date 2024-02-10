import cv2
import os
from ultralytics import YOLO
import numpy as np 



# Set the environment variable for OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
cap = cv2.VideoCapture(r"C:\Users\LapTop\OneDrive\Desktop\High key ethereal fog blows over people on sandy ocean beach, Ireland (1).mp4")

# Get the original width and height of the video
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the desired width of the displayed window
desired_width = 800

# Calculate the aspect ratio
aspect_ratio = original_width / original_height

# Calculate the new height based on the desired width and aspect ratio
desired_height = int(desired_width / aspect_ratio)

# Initialize background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=5, detectShadows=False) 


# Loop through the video frames
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # If frame is read correctly, proceed
    if ret:
        # Resize the frame while preserving the aspect ratio
        frame = cv2.resize(frame, (desired_width, desired_height))

        # Apply background subtraction
        fg_mask = bg_subtractor.apply(frame)
        # 3 channel for the model 
        fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(fg_mask, persist=True)

        # Check if there are any detections
        if results[0].boxes is not None:
            # Iterate through the detected objects
            for i, box in enumerate(results[0].boxes.xyxy.cpu().numpy().astype(int)):
                # Check if the object is a person
                if results[0].names[results[0].boxes.cls[i].item()][0] == 'person'[0]:
                    print("Number of persons detected:", len(results[0].boxes))
                    break
        
        # Display the motion mask
        cv2.imshow("Motion Mask", fg_mask)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached or frame is not read correctly
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
