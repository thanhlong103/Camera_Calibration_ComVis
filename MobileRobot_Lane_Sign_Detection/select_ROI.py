import cv2 as cv 
import numpy as np

frame_shape = [640, 480]
cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv.resize(frame,(int(frame_shape[0]/2), int(frame_shape[1]/2)))

    # Select ROI
    r = cv.selectROI("select the area", frame)
    # Crop frame
    cropped_frame = frame[int(r[1]):int(r[1]+r[3]),
                        int(r[0]):int(r[0]+r[2])]
    print(r)
# Display cropped image
cv2.imshow("Cropped image", cropped_frame)
cv2.waitKey(0)
