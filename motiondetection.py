import cv2
import numpy as np
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent
from sklearn.decomposition import PCA
import scipy
print(scipy.__version__)


# Function to apply RPCA to a single frame
def apply_rpca(frame):
    # Reshape the frame to a 2D matrix
    X = frame.reshape(-1, frame.shape[2])

    # Apply PCA to obtain low-rank and sparse components
    pca = PCA(n_components=1)
    low_rank = pca.fit_transform(X)
    sparse = X - pca.inverse_transform(low_rank)

    # Reshape the components back to the original frame shape
    low_rank_frame = low_rank.reshape(frame.shape)
    sparse_frame = sparse.reshape(frame.shape)

    return low_rank_frame, sparse_frame

# Open the video file
cap = cv2.VideoCapture('input_video.mp4')

# Initialize video writer for output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (640, 480))

# Loop through the video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Apply RPCA to the frame
    low_rank_frame, sparse_frame = apply_rpca(frame)

    # Display the original frame, low-rank component, and sparse component
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Low-Rank Component', low_rank_frame)
    cv2.imshow('Sparse Component', sparse_frame)

    # Write the low-rank component to the output video
    out.write(np.uint8(low_rank_frame))

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
