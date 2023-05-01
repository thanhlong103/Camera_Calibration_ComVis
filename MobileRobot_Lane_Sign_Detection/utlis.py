########## LIBRARIES ##########
import cv2 as cv
import numpy as np
import math
from imutils.perspective import four_point_transform
from imutils import contours
import imutils



########## PARAMETERS FOR TUNING ##########

### Tune Canny edge detection ###
desired_edge_number = 10

frame_shape = [320, 240]

### Tune range HSV for blue color of the traffic sign ###
lower_blue = np.array([85,100,70])
upper_blue = np.array([125,255,255])

### Tune range HSV for blue color of the traffic sign ###
lower_red = np.array([150,25,0])
upper_red = np.array([10,255,220])



########## HELPER FUNCTIONS ##########

### FOR LANE DETECTION ###

# Function to generate an adaptive low_threshold when applying Canny Edge Detection
def canny_adaptiveThresh(blurred_image, line_number):
    thresh = 0
    while True:
        thresh += 10
        canny_image = cv.Canny(blurred_image, thresh, thresh*3, None, 3) # high_threshold = low_threshold * 3
        contours, hierarchy = cv.findContours(canny_image.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        number_lines = len(contours)
        if number_lines == 0:
            thresh == 0
            continue
        if number_lines <= line_number:
            break
    return canny_image

# Function to generate an adaptive threshold when apply Hough Line Transform on processed image
def houghLines_adaptiveThresh(processed_image):
    thresh = 200
    const = 0
    while const == 0:
        thresh -= 10
        hough_transform_lines = cv.HoughLines(processed_image, 1, np.pi / 180, thresh, None, 0, 0)
        pos = 0
        neg = 0
        if hough_transform_lines is None:
            continue
        if hough_transform_lines is not None:
            for i in range(len(hough_transform_lines)):
                if pos != 0 and neg != 0:
                    break
                if hough_transform_lines[i][0][0] > 0:
                    pos += 1
                if hough_transform_lines[i][0][0] < 0:
                    neg += 1
        if pos and neg != 0:
            break
    rho_max = 0
    rho_min = 0
    if len(hough_transform_lines) > 2:
        for i in range(len(hough_transform_lines)):
            if hough_transform_lines[i][0][0] > 0:
                if hough_transform_lines[i][0][0] > rho_max:
                    rho_max = hough_transform_lines[i][0][0]
                    index_rho_max = i
            else:
                if hough_transform_lines[i][0][0] < rho_min:
                    rho_min = hough_transform_lines[i][0][0]
                    index_rho_min = i
        filtered_hough_transform_lines = np.array([hough_transform_lines[index_rho_max], hough_transform_lines[index_rho_min]])
    else:
        filtered_hough_transform_lines = hough_transform_lines.copy()
    return filtered_hough_transform_lines

# Function to convert the input image to gray scale, apply blur, extract edges, and dilate the edged image
def image_processing(inputImage):
    gray = cv.cvtColor(inputImage, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    canny = canny_adaptiveThresh(blur, desired_edge_number)
    dilate = cv.dilate(canny, (1, 1), iterations=3)
    return dilate

# Function to draw the lines on BGR image
def line_drawing(hough_transform_lines, image):
    for i in range(0, len(hough_transform_lines)):
        rho = hough_transform_lines[i][0][0]
        theta = hough_transform_lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        drawn_image = cv.line(image, pt1, pt2, (0, 255, 0), 3, cv.LINE_AA)
    return drawn_image

# For each line in line_drawn_BGR_cvtImg -> (theta, rho)
# rho = x*cos(theta) + y*sin(theta)
# line equation: Ax + By + C = 0
# A = cos(theta), B = sin(theta), C = -rho
def line_coefficients(line):
    A = math.cos(line[0][1])
    B = math.sin(line[0][1])
    C = -line[0][0]
    list_lineCoefficients = np.array([A, B, C])
    return list_lineCoefficients

# Get variables x, y of the bisector's line equation: Ax + By + C = 0 => x = (-C - By) / A
def bisector_point(bisector_coefficients, y):
    x = ((-bisector_coefficients[2])-(bisector_coefficients[1]*y))/bisector_coefficients[0]
    if x > frame_shape[0]:
        x = frame_shape[0]
    point = (int(x), int(y))
    return point

# Function to draw a midline in the input image
def midline_drawing(inputImage):
    # Process the input image
    processed_image = image_processing(inputImage.copy())

    # Hough Line Transform with self-built adaptive-threshold Hough Line Transform function
    hough_transform_lines = houghLines_adaptiveThresh(processed_image)

    # Draw the lines
    line_drawn_image = line_drawing(hough_transform_lines, inputImage.copy())

    # Get line equation's coefficients of drawn lines (Ax + By + C)
    line1_coefficients = line_coefficients(hough_transform_lines[0])
    line2_coefficients = line_coefficients(hough_transform_lines[1])

    # Get line equation's coefficients for the bisector line
    bisector_coefficients = line1_coefficients - line2_coefficients

    # Get two points lied on the bisector line
    bisector_point1 = bisector_point(bisector_coefficients, 0)
    bisector_point2 = bisector_point(bisector_coefficients, frame_shape[1])

    # Draw the bisector line
    midline_drawn_image = cv.line(line_drawn_image.copy(), bisector_point1, bisector_point2, (0,0,255), 4, cv.LINE_AA)

    return midline_drawn_image

# Function to get bisector points of two lanes on two sides
def get_midline_points(inputImage):
    # Process the input image
    processed_image = image_processing(inputImage.copy())
    # Hough Line Transform with self-built adaptive-threshold Hough Line Transform function
    hough_transform_lines = houghLines_adaptiveThresh(processed_image)
    if hough_transform_lines is not None:
        # Draw the lines
        line_drawn_image = line_drawing(hough_transform_lines, inputImage.copy())
        # Get line equation's coefficients of drawn lines (Ax + By + C)
        line1_coefficients = line_coefficients(hough_transform_lines[0])
        line2_coefficients = line_coefficients(hough_transform_lines[1])
        # Get line equation's coefficients for the bisector line
        bisector_coefficients = line1_coefficients - line2_coefficients
        # Get two points lied on the bisector line
        bisector_point1 = bisector_point(bisector_coefficients, 0)
        bisector_point2 = bisector_point(bisector_coefficients, frame_shape[1])
        result = [bisector_point1, bisector_point2]
    return result

# Function to remove outliers when detecting the bisector points
def filter_midpoint(midpoint_list):
    an_array = np.array(midpoint_list)
    mean = np.mean(an_array)
    standard_deviation = np.std(an_array)
    if standard_deviation == 0:
        standard_deviation = 1
    distance_from_mean = abs(an_array - mean)
    max_deviations = 2
    not_outlier = distance_from_mean < max_deviations * standard_deviation
    no_outliers = an_array[not_outlier]
    result = np.mean(no_outliers)
    return result


### FOR SIGN DETECTION ###

def findTrafficSign(frame):
    # This function find blobs with blue color on the image.
    # After blobs were found it detects the largest square blob, that must be the sign.
    
    frameArea = frame.shape[0]*frame.shape[1]
    
    # convert color image to HSV color scheme
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    # define kernel for smoothing   
    kernel = np.ones((3,3),np.uint8)

    # extract binary image with active blue regions
    mask1 = cv.inRange(hsv, lower_blue, upper_blue)
    mask2 = cv.inRange(hsv, lower_red, upper_red)

    # morphological operations
    mask1 = cv.morphologyEx(mask1, cv.MORPH_OPEN, kernel)
    mask1 = cv.morphologyEx(mask1, cv.MORPH_CLOSE, kernel)

    mask2 = cv.morphologyEx(mask2, cv.MORPH_OPEN, kernel)
    mask2 = cv.morphologyEx(mask2, cv.MORPH_CLOSE, kernel)

    # find contours in the mask
    cnts1 = cv.findContours(mask1.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
    cnts2 = cv.findContours(mask2.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
    
    # defite string variable to hold detected sign description
    detectedTrafficSign = None
    
    # define variables to hold values during loop
    largestArea = 0
    largestRect = None
    
    # only proceed if at least one contour was found
    if len(cnts1) > 0:
        for cnt in cnts1:

            # Rotated Rectangle. Here, bounding rectangle is drawn with minimum area,
            # so it considers the rotation also. The function used is cv.minAreaRect().

            # It returns a Box2D structure which contains following detals -
            # (center (x,y), (width, height), angle of rotation ).

            # But to draw this rectangle, we need 4 corners of the rectangle.
            # It is obtained by the function cv.boxPoints()

            rect = cv.minAreaRect(cnt)
            box = cv.boxPoints(rect)
            box = np.int0(box)

            # Count euclidian distance for each side of the rectangle
            sideOne = np.linalg.norm(box[0]-box[1])
            sideTwo = np.linalg.norm(box[0]-box[3])

            # Count area of the rectangle
            area = sideOne*sideTwo

            # Find the largest rectangle within all contours
            if area > largestArea:
                largestArea = area
                largestRect = box

    elif len(cnts2)>2:
        for cnt in cnts2:
            rect = cv.minAreaRect(cnt)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            
            sideOne = np.linalg.norm(box[0]-box[1])
            sideTwo = np.linalg.norm(box[0]-box[3])

            area = sideOne*sideTwo

            if area > largestArea:
                largestArea = area
                largestRect = box          

    # Draw contour of the found rectangle on  the original image
    if largestArea > frameArea*0.02:
        cv.drawContours(frame,[largestRect],0,(0,0,255),2)
        

    if largestRect is not None:
        # cut and warp interesting area
        warped = four_point_transform(mask1, [largestRect][0])
        
        # use function to detect the sign on the found rectangle
        if len(cnts1) > 0:
            detectedTrafficSign = identifyTrafficSign(warped)
        elif len(cnts2) >0:
            detectedTrafficSign = identifyTrafficSign_red(warped)

        # write the description of the sign on the original image
        cv.putText(frame, detectedTrafficSign, tuple(largestRect[0]), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    return detectedTrafficSign 

   
def identifyTrafficSign(image):

    # In this function we select some ROI in which we expect to have the sign parts. If the ROI has more active pixels than threshold we mark it as 1, else 0
    # After path through all four regions, we compare the tuple of ones and zeros with keys in dictionary SIGNS_LOOKUP

    # Define the dictionary of sign's segments so we can identify
    # Each sign on the image
    SIGNS_LOOKUP = {
        (1, 0, 0, 1): 'RIGHT', # turnRight
        (0, 0, 1, 1): 'LEFT', # turnLeft
        (0, 1, 0, 1): 'RIGHT',
        (1, 0, 1, 1): 'STOP', # Stop
    }

    THRESHOLD = 150
    
    image = cv.bitwise_not(image)
    (subHeight, subWidth) = np.divide(image.shape, 10)
    subHeight = int(subHeight)
    subWidth = int(subWidth)

    # mark the ROIs borders on the image
    cv.rectangle(image, (subWidth, 4*subHeight), (3*subWidth, 9*subHeight), (0,255,0),2) # left block
    cv.rectangle(image, (4*subWidth, 4*subHeight), (6*subWidth, 9*subHeight), (0,255,0),2) # center block
    cv.rectangle(image, (7*subWidth, 4*subHeight), (9*subWidth, 9*subHeight), (0,255,0),2) # right block
    cv.rectangle(image, (2*subWidth, 2*subHeight), (7*subWidth, 4*subHeight), (0,255,0),2) # top block

    # substract 4 ROI of the sign thresh image
    leftBlock = image[4*subHeight:9*subHeight, subWidth:3*subWidth]
    centerBlock = image[4*subHeight:9*subHeight, 4*subWidth:6*subWidth]
    rightBlock = image[4*subHeight:9*subHeight, 7*subWidth:9*subWidth]
    topBlock = image[2*subHeight:4*subHeight, 3*subWidth:7*subWidth]

    # we now track the fraction of each ROI
    leftFraction = np.sum(leftBlock)/(leftBlock.shape[0]*leftBlock.shape[1])
    centerFraction = np.sum(centerBlock)/(centerBlock.shape[0]*centerBlock.shape[1])
    rightFraction = np.sum(rightBlock)/(rightBlock.shape[0]*rightBlock.shape[1])
    topFraction = np.sum(topBlock)/(topBlock.shape[0]*topBlock.shape[1])

    segments = (leftFraction, centerFraction, rightFraction, topFraction)
    segments = tuple(1 if segment > THRESHOLD else 0 for segment in segments)

    if segments in SIGNS_LOOKUP:
        return SIGNS_LOOKUP[segments]
    else:
        return None


def identifyTrafficSign_red(image):

    # In this function we select some ROI in which we expect to have the sign parts. If the ROI has more active pixels than threshold we mark it as 1, else 0
    # After path through all four regions, we compare the tuple of ones and zeros with keys in dictionary SIGNS_LOOKUP

    # Define the dictionary of signs segments so we can identify
    # Each signs on the image
    THRESHOLD = 150
    
    image = cv.bitwise_not(image)
    (subHeight, subWidth) = np.divide(image.shape, 10)
    subHeight = int(subHeight)
    subWidth = int(subWidth)

    # Mark the ROIs borders on the image
    cv.rectangle(image, (subWidth, 4*subHeight), (3*subWidth, 9*subHeight), (0,255,0),2) # left block
    cv.rectangle(image, (4*subWidth, 4*subHeight), (6*subWidth, 9*subHeight), (0,255,0),2) # center block
    cv.rectangle(image, (7*subWidth, 4*subHeight), (9*subWidth, 9*subHeight), (0,255,0),2) # right block
    cv.rectangle(image, (2*subWidth, 2*subHeight), (7*subWidth, 4*subHeight), (0,255,0),2) # top block

    return "STOP"