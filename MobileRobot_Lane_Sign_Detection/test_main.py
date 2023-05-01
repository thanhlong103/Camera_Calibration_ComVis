########## LIBRARIES ##########
import cv2 as cv
import numpy as np
import utlis
import serial


########## PARAMETERS ##########

### Open communication between Raspberry Pi and Arduino ###
port = '/dev/ttyAMA0'
#port = '/dev/ttyUSB0'
# arduino = serial.Serial(port, 9600, timeout=1)

### Original shape before resizing ###
frame_shape = [640, 480] 

### Region of interest for lane detection ###
roi_lane = [3, 142, 317, 98] #x1,y1,x2,y2

### Region of interest for sign detection ###
roi_sign = [85, 1, 158, 134] #x1,y1,x2,y2

### Define ranges to control robot's moving direction ###
midpoint = int(roi_lane[2]-roi_lane[0]) / 2
std = 20
range1 = [midpoint-std, midpoint+std]
range2 = [range1[0]-std, range1[1]+std]
range3 = [range2[0]-std, range2[1]+std]

### Number of midpoints calculated that need removing outliers ###
number_midpoint_forFilter = 10


########## MAIN LOOP ##########
if __name__ == '__main__':
    cap = cv.VideoCapture(0)
    
    while True:
        
        ### PROCESSING LANE DETECTION ###
        top_midpoint_list = []
        bottom_midpoint_list = []

        while len(top_midpoint_list) <= number_midpoint_forFilter:
        # The while loop aims at collecting 10 recent consecutive bisector points
        # and applying a filter to remove outliers before sending moving commands
        # to the Arduino

            ret, frame = cap.read()

            if frame[0][0][0] == 0: # To avoid black frames of webcam at launch
                continue

            frame = cv.resize(frame,(int(frame_shape[0]/2), int(frame_shape[1]/2))) # Resize the origital frame shape by half

            roi_lane_frame = frame[int(roi_lane[1]):int(roi_lane[1]+roi_lane[3]), # Crop the ROI for detecting lane
                                int(roi_lane[0]):int(roi_lane[0]+roi_lane[2])]

            midpoint_list = utlis.get_midline_points(roi_lane_frame) # Get 2 bisector points of 2 detected lanes on 2 sides
            # print("midpoint list", midpoint_list)

            top_midpoint = midpoint_list[0][0]
            top_midpoint_list.append(top_midpoint)

            bottom_midpoint = midpoint_list[1][0]
            bottom_midpoint_list.append(bottom_midpoint)

            processed_frame = utlis.midline_drawing(roi_lane_frame)
            cv.imshow("ROI lane", processed_frame)
            cv.waitKey(1)


            ### SIGN DETECTION ###
            roi_sign_frame = frame[int(roi_sign[1]):int(roi_sign[1]+roi_sign[3]), # Crop the ROI for detecting signs
                                int(roi_sign[0]):int(roi_sign[0]+roi_sign[2])]

            sign = utlis.findTrafficSign(roi_sign_frame)
            # CASES for SIGN DETECTION
            if sign is not None:
                if sign == "STOP":
                    # arduino.write(b'j')
                    print('j')
                elif sign == "RIGHT":
                    # arduino.write(b'k')
                    print('k')
                elif sign == "LEFT":
                    # arduino.write(b'l')
                    print('l')

            ## WINDOW DISPLAY ##
            cv.imshow("Sign detection", roi_sign_frame)
            cv.waitKey(1)

        ## Remove outliers among the 10 most recent collected bisector points ##
        filter_top_midpoint = (int(utlis.filter_midpoint(top_midpoint_list)), 0)
        filter_bottom_midpoint = (int(utlis.filter_midpoint(bottom_midpoint_list)), int(frame_shape[1]/2))
        print("top: ", filter_top_midpoint)
        print("bottom: ", filter_bottom_midpoint)
        top = int(filter_top_midpoint[0])
        bottom = int(filter_bottom_midpoint[0])

        ## WINDOW DISPLAY ##
        filter_drawn_frame = cv.line(processed_frame.copy(), filter_bottom_midpoint, filter_top_midpoint, (255,0,255), 4, cv.LINE_AA)
        cv.imshow('filter FRAME', filter_drawn_frame)
        cv.waitKey(1)

        # CASES for LANE DETECTION
        if range1[0] <= top < range1[1]:
            # arduino.write(b'a')
            print('a')
        elif range2[0] <= top < range1[0]:
            # arduino.write(b'b')
            print('b')
        elif range3[0] < top < range2[0]:
            # arduino.write(b'c')
            print('c')
        elif range1[1] <= top < range2[1]:
            # arduino.write(b'd')
            print('d')
        elif range2[1] <= top < range3[1]:
            # arduino.write(b'e')
            print('e')
        elif top >= range3[1]:
            # arduino.write(b'f')
            print('f')
        elif top >= range3[1] and bottom >= range3[1]:
            # arduino.write(b'g')
            print('g')
        elif top <= range3[0]:
            # arduino.write(b'h')
            print('h')
        elif top <= range3[0] and bottom <= range3[0]:
            # arduino.write(b'i')
            print('i')