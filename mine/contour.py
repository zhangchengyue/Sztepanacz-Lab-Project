import cv2 as cv
import numpy as np
import os

# 完整cv2的教学：https://www.youtube.com/watch?v=oXlwWbU8l2o
# https://www.youtube.com/watch?v=IBQYqwq_w14

# Absolute path of the image
path = os.path.expanduser("~/St.George/2022Summer/WorkStudy/Project/mine/So05D_16.jpg")

# Read image
img = cv.imread(path)

# Show image in a different window
cv.imshow('Original', img)


# Resize and rescale image
def rescaleFrame(frame, scale=0.75):
    """
    :param frame: Input frame
    :param scale: Scales the frame by a particular value, default of 0.75
    :return: New frame
    """
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


# Show resized image
frame_resized = rescaleFrame(img)  # default scale = 0.75
# cv.imshow('Resized', frame_resized)


# Create blank image
blank = np.zeros((500, 500, 3), dtype='uint8')  # The first tuple contains (height, width, number of color channels)
# cv.imshow('Blank', blank)

# 1. Paint the image a certain colour
# Some commonly used colors:
# 0, 255, 0: Green
# 255, 255, 0: Blue
# 0, 0, 255: Red
# 0, 255, 255: Yellow
# 255, 0, 255: Purple
# 0, 0, 0: Black
# 255, 255, 255: White
green = blank.copy()
green[:] = 0, 255, 0  # Green
# cv.imshow('Green', green)

red = blank.copy()
red[:] = 0, 0, 255  # Red
# cv.imshow('Red', red)

portion = blank.copy()
portion[100:200, 200:400] = 255, 255, 0  # Blue
# cv.imshow('Portions', portion)

# 2. Draw a rectangle
rec = blank.copy()
cv.rectangle(rec, (0, 0), (255, 255), (0, 255, 0), thickness=2)
# cv.rectangle parameters:
# 1) The image to be drawn on
# 2) Start point
# 3) End point
# 4) Color of the line to be drawn
# 5) Thickness of the line. If you want to fill in the space, use cv.FILLED

# cv.imshow('Rectangle', rec)


# 3. Draw a circle
circle = blank.copy()
cv.circle(circle, (250, 250), 40, (0, 0, 255), thickness=3)
# 1) Input image
# 2) Center of the circle
# 3) Radius
# 4) Colour
# 5) Thickness

# cv.imshow('Circle', circle)

# 4. Draw a line
line = blank.copy()
cv.line(line, (204, 144), (644, 144), (255, 255, 255), thickness=3)
cv.imshow('Line', line)


# 5. Write or Text on an image. Used Hershey Triplex for font type in this example, search for more if needed.
text = blank.copy()
cv.putText(text, 'Hello World!', (150, 250), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), thickness=2)
# 1) Image to put text on
# 2) The text you want to write
# 3) Position of the text
# 4) Font type
# 5) Font size
# 6) Color of text
# 7) Thickness of text

# cv.imshow('Text', text)


# Converting to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)


# Blur an image to remove some noise. Use Gaussian in this example, search for more if needed.
blur = cv.GaussianBlur(img, (3, 3), cv.BORDER_DEFAULT)
# 1) Image to be blurred
# 2) Kernel size. Must be a tuple of 2 odd numbers, eg. (3, 3) or (9, 9)
# 3) Border
# cv.imshow('Blur', blur)


# Find the edges that are present in the image. The edge cascade used this example is Canny, search for more if needed
canny = cv.Canny(blur, 85, 0)
# 1) Input image
# 2) & 3) Two thresholds
# cv.imshow('Canny Edges', canny)


# Dilating the image
dilated = cv.dilate(canny, (3, 3), iterations=1)
# cv.imshow('Dilated', dilated)

# Eroding
eroded = cv.erode(dilated, (3, 3), iterations=1)
# cv.imshow('Eroded', eroded)

# Resize
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_AREA)
# cv.imshow('Resized', resized)

# Cropping
cropped = img[500:700, 700:900]
# cv.imshow('Cropped', cropped)


gray_resized = rescaleFrame(gray)
ret, thresh = cv.threshold(gray_resized, 125, 255, 0)
contours, hierachy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
cv.drawContours(frame_resized, contours, -100, (0, 255, 0), thickness=6)
print('Total of ' + str(len(contours)) + 'contours.')
print(contours)
cv.imshow('Draw contours', frame_resized)
# cv.imwrite(os.path.expanduser("~/St.George/2022Summer/WorkStudy/Project/mine/contour_") + path[-12:], img)


# Waits for infinite time until a key is pressed
cv.waitKey(0)
