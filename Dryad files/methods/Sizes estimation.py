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
path = os.path.expanduser("~/St.George/2022Summer/WorkStudy/Project/So05D_16.jpg")
upper_x = []
upper_y = []

lower_x = []
lower_y = []


with open(os.path.expanduser("~/St.George/2022Summer/WorkStudy/Project/landmarks_plasticity.txt"), 'r') as f:
    for line in f.readlines():
        l = line.strip().split(',')
        if l[0] == path[-12:-4]:
            upper_x = l[1]
            upper_y = l[2]
            lower_x = l[3]
            lower_y = l[4]

im = cv2.imread(path)

height, width, channels = im.shape

mask = np.zeros((height, width), np.uint8)

if path[-8] == 'D':
    pts = np.array([[0, 0], [0, height], [lower_x, lower_y], [upper_x, upper_y]])

if path[-8] == 'G':
    pts = np.array([[upper_x, upper_y], [lower_x, lower_y], [width, height], [width, 0]])

_ = cv2.drawContours(mask, np.int32([pts]), 0, 255, -1)

raw_wing = im.copy()

raw_wing[mask > 0] = 255  # cut area will be white (255)

cv2.line(raw_wing, (int(upper_x), int(upper_y)), (int(lower_x), int(lower_y)),
         (255, 0, 0), thickness=8, lineType=cv2.LINE_AA)

# We detect the wing contour by setting up a colour threshold

gray = cv2.cvtColor(raw_wing, cv2.COLOR_BGR2GRAY)

dark = cv2.equalizeHist(gray)

ret, thresh = cv2.threshold(dark, 90, 255, cv2.THRESH_BINARY_INV)

_, cont, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

ith_largest, cont_index = contour_index(cont, 0)

cv2.drawContours(im, cont, cont_index, (255, 255, 0), 6)

cv2.namedWindow('Disp', cv2.WINDOW_NORMAL)

cv2.imshow('Disp', im)

areaWing = cv2.contourArea(ith_largest)

# Then the spot contour is estimated

blurred = cv2.pyrMeanShiftFiltering(im, 31, 91)  # dilate & erode doesn't work well

gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

retval, threshold = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)

_, contours, _ = cv2.findContours(threshold, cv2.RETR_LIST,
                                  cv2.CHAIN_APPROX_NONE)

spot_ith_larg, spot_cont_ind = contour_index(contours, 1)

cv2.drawContours(im, contours, spot_cont_ind, (255, 255, 0), 6)

cv2.imshow('Disp', im)

areaSpot = cv2.contourArea(spot_ith_larg)

# Estimatie Spot / wing ratio and save results

ratio = areaSpot / areaWing

cv2.imwrite(str("~/St.George/2022Summer/WorkStudy/Project/Spot_Area") + path[-12:], im)

with open("~/St.George/2022Summer/WorkStudy/Project/results.txt", 'a') as new_results:
    new_results.write(str(path[-12:-8]) + ',' + str(path[-20:-13]) + ',' +
                      str(path[-6:-4]) + ',' + str(path[-8]) + ',' + str(areaWing)
                      + ',' + str(areaSpot) + ',' + str(ratio) + '\n')
