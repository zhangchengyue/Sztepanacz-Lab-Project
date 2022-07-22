import os

import cv2
import numpy as np


def contour_index(contours, ith_bigger):
    """Finds the ith biggest contour among a set of contours.

    Args:
        contours: a set of contours as obtained from cv2.findContours
        ith_bigger: 0 for the biggest, 1 for the 2nd biggest, etc

    Returns:
        The ith biggest contour and its index in the original set of contours
    """
    areaArray = []
    count = 1
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        areaArray.append(area)
    sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)
    ith_largest_contour = sorteddata[ith_bigger][1]
    cont_index = areaArray.index(cv2.contourArea(ith_largest_contour))
    return (ith_largest_contour, cont_index)


# We start by cutting out the part of the picture where the thorax is

# This is the path of the picture for windows system
# path = str('C:\\Users\\vllaurens\\Desktop\\Spot_images\\SOKOL16\\So05D_16.jpg')

# This is the path of the picture for IOS system
path = os.path.expanduser("~/St.George/2022Summer/WorkStudy/Project/So05G_16.tif")

# Coordinators for landmarks
# upper_x = []
# upper_y = []
#
# lower_x = []
# lower_y = []

upper_x = '2300'
upper_y = '400'

lower_x = '2300'
lower_y = '800'
# Read landmarks from file.
# TODO: Try to use ImageJ to detect landmarks for each picture, and then save the coordinates in a new file.
# with open(os.path.expanduser("~/St.George/2022Summer/WorkStudy/Project/landmarks_plasticity.txt"), 'r') as f:
#     for line in f.readlines():
#         l = line.strip().split(',')
#         if l[0] == path[-12:-4]:
#             upper_x = l[1]
#             upper_y = l[2]
#             lower_x = l[3]
#             lower_y = l[4]

im = cv2.imread(path)

# Return the dimension of the given image as a tuple
# tuple[0]: height of image
# tuple[1]: width of image
# tuple[2]: number of channels (colors) in the image.
# eg. RGB image has 3 channels, Grayscale image has only 1 channel
height, width, channels = im.shape

# Creates a black image with the same size of the wing image
mask = np.zeros((height, width), np.uint8)

# Uncomment the following to look at the mask
# cv2.imshow('Numpy.zero', mask)
# cv2.waitKey(0)

# Creates a shape with landmarks as boundary. The shape is represented as numpy array
if path[-8] == 'D':
    pts = np.array([[0, 0], [0, height], [lower_x, lower_y], [upper_x, upper_y]])

if path[-8] == 'G':
    pts = np.array([[upper_x, upper_y], [lower_x, lower_y], [width, height], [width, 0]])

print('Upper landmark: (' + upper_x + ', ' + upper_y + ')')
print('Lower landmark: (' + lower_x + ', ' + lower_y + ')')

# Draw contours of the shape bounded by landmarks
_ = cv2.drawContours(mask, np.int32([pts]), 0, 255, -1)

# Uncomment the following to look at the mask with landmarks drawn
# cv2.imshow('Numpy.zero', mask)
# cv2.waitKey(0)


raw_wing = im.copy()

# Cut out the unwanted part
raw_wing[mask > 0] = 255  # cut area will be white (255)

cv2.line(raw_wing, (int(upper_x), int(upper_y)), (int(lower_x), int(lower_y)),
         (255, 0, 0), thickness=8, lineType=cv2.LINE_AA)

# Uncomment the following to look at raw_wing
# cv2.imshow('raw_wing', raw_wing)
# cv2.waitKey(0)

# We detect the wing contour by setting up a colour threshold

# Turn BGR image to grayscale
gray = cv2.cvtColor(raw_wing, cv2.COLOR_BGR2GRAY)

# Returns a histogram equalized image.
# Improves the contrast in the image
dark = cv2.equalizeHist(gray)

# Uncomment the following to look at histogram equalized image
# cv2.imshow('Histogram equalized image', dark)
# cv2.waitKey(0)


# Flip black and white colors in the image
# ret, thresh = cv2.threshold(dark, 90, 255, cv2.THRESH_BINARY_INV)
ret, thresh = cv2.threshold(dark, 90, 255, cv2.THRESH_BINARY_INV)

# Uncomment the following to look at thresh
# cv2.imshow('Threshold', thresh)
# cv2.waitKey(0)


# Find all sets of contours in the image
cont, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# Find the largest contour
ith_largest, cont_index = contour_index(cont, 0)

cv2.drawContours(im, cont, cont_index, (255, 255, 0), 6)

# Uncomment the following to look at the largest contour of the image
cv2.imshow('Largest contour', im)
cv2.waitKey(0)

cv2.namedWindow('Disp', cv2.WINDOW_NORMAL)

cv2.imshow('Disp', im)

areaWing = cv2.contourArea(ith_largest)

# Then the spot contour is estimated
blurred = cv2.pyrMeanShiftFiltering(im, 31, 91)  # dilate & erode doesn't work well

gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

retval, threshold = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(threshold, cv2.RETR_LIST,
                               cv2.CHAIN_APPROX_NONE)

# It is estimated that the second-largest contour would be the wing spot
spot_ith_larg, spot_cont_ind = contour_index(contours, 1)

cv2.drawContours(im, contours, spot_cont_ind, (255, 255, 0), 6)

cv2.imshow('Disp', im)

areaSpot = cv2.contourArea(spot_ith_larg)

# Estimatie Spot / wing ratio and save results

ratio = areaSpot / areaWing

print('The ratio is: {}'.format(ratio))
# cv2.imwrite(str("~/St.George/2022Summer/WorkStudy/Project/Spot_Area") + path[-12:], im)
#
# with open("~/St.George/2022Summer/WorkStudy/Project/results.txt", 'a') as new_results:
#     new_results.write(str(path[-12:-8]) + ',' + str(path[-20:-13]) + ',' +
#                       str(path[-6:-4]) + ',' + str(path[-8]) + ',' + str(areaWing)
#                       + ',' + str(areaSpot) + ',' + str(ratio) + '\n')

# TODO: Get the size of reference line, and measure the object size.
