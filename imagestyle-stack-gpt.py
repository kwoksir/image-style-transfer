import cv2
import numpy as np
from cvzone import *

# Load the style transfer models
models = [
    cv2.dnn.readNetFromTorch('model/starry_night.t7'),
    cv2.dnn.readNetFromTorch('model/the_scream.t7'),
    cv2.dnn.readNetFromTorch('model/mosaic.t7'),
    cv2.dnn.readNetFromTorch('model/the_wave.t7'),
    cv2.dnn.readNetFromTorch('model/candy.t7'),
    cv2.dnn.readNetFromTorch('model/feathers.t7'),
    cv2.dnn.readNetFromTorch('model/udnie.t7'),
    cv2.dnn.readNetFromTorch('model/la_muse.t7'),
    #cv2.dnn.readNetFromTorch('model/la_muse-norm.t7'),
    cv2.dnn.readNetFromTorch('model/composition_vii.t7'),
]

# Define the blending factor
alpha = 0.5

# Initialize the video capture
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    
    # Resize the frame
    frame = cv2.resize(frame, (480, 360))
    
    # Apply style transfer to the frame using each model
    stylized_frames = []
    for model in models:
        blob = cv2.dnn.blobFromImage(frame, 1.0, (480, 360), (103.939, 116.779, 123.680), swapRB=False, crop=False)
        model.setInput(blob)
        output = model.forward()
        output = output.reshape((3, output.shape[2], output.shape[3]))
        output[0] += 103.939
        output[1] += 116.779
        output[2] += 123.680
        output /= 255.0
        output = output.transpose(1, 2, 0)
        stylized_frames.append(output)
    stacked_frames = stackImages(stylized_frames,3,1)
    # Stack the stylized frames and display them
    #stacked_frames = np.hstack(stylized_frames)
    cv2.imshow('Stylized Frames', stacked_frames)
    
    # Blend the stylized frame with the original frame and display them
    #blended_frame = alpha * stylized_frames[0] + (1 - alpha) * frame
    cv2.imshow('Normal Frame', frame)
    
    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
