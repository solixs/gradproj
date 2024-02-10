import cv2
import os
import numpy as np

# Set the environment variable for OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Open the video file
cap = cv2.VideoCapture("C:\motion\WhatsApp Video 2024-02-04 at 3.08.48 PM.mp4")

# Initialize background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=5, detectShadows=True) 

# Structuring element for morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

# Minimum contour area threshold
min_contour_area = 50

# Loop through the video frames
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # If frame is read correctly, proceed
    if ret:
        # Apply background subtraction
        fg_mask = bg_subtractor.apply(frame)

        # Threshold the foreground mask to identify pixels with variance within the specified range
        _, thresholded_mask = cv2.threshold(fg_mask, 10, 20, cv2.THRESH_BINARY)

        # Fill the gaps between connected foreground pixels using morphological operations
        filled_mask = cv2.dilate(thresholded_mask, kernel, iterations=4)
        filled_mask = cv2.erode(filled_mask, kernel, iterations=4)

        # Create a blank image to draw filtered contours
        filtered_contour_img = np.ones_like(filled_mask) * 255

        # Find contours in the filled mask
        contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

        # Draw filtered contours on the blank image
        cv2.drawContours(filtered_contour_img, filtered_contours, -1, (0, 0, 0), thickness=-1)

        # Plot the areas where the object is not moving but is part of the moving object
        stationary_areas = cv2.add(filled_mask, cv2.bitwise_not(filtered_contour_img))

        # Display the stationary areas
        cv2.imshow("Stationary Areas", stationary_areas)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached or frame is not read correctly
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()