import os

import cv2
import imutils
import numpy as np
from imutils import perspective
from scipy.spatial import distance as dist


def midpoint(ptA, ptB):
    """
    Detect the midpoint of two coordinators.
    Returns the coordinator of the midpoint.
    """
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


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


def compute_size(im, c, pixel, name):
    """
    Compute the size of the object
    :param im: Input image
    :param c: Contour of the object
    :param pixel: Pixels per Metric used as reference
    :param name: Name of this object
    :return: A tuple (height, width, pixels per metric)
    """
    orig = im.copy()
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

    # Unpack the ordered bounding box, then compute the midpoint
    # between the top-left and top-right coordinates, followed by
    # the midpoint between bottom-left and bottom-right coordinates
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    # Compute the midpoint between the top-left and bottom-left points,
    # followed by the midpoint between the top-right and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # Draw the midpoints on the image
    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
    # draw lines between the midpoints
    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
             (255, 0, 255), 2)
    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
             (255, 0, 255), 2)

    # Compute the Euclidean distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    # If the pixels per metric has not been initialized, then
    # compute it as the ratio of pixels to supplied metric
    # (in this case, nm)
    if pixel is None:
        pixel = dB / 500

    # compute the size of the object
    width = dA / pixel
    length = dB / pixel
    # draw the object sizes on the image
    cv2.putText(orig, "{:.1f} um".format(width),
                (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    cv2.putText(orig, "{:.1f} um".format(length),
                (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    cv2.imshow("{}".format(name), orig)
    print('{} size :\n'.format(name) + 'Length: {:.2f} um \n'.format(length) +
          'Width: {:.2f} um \n'.format(width) + '~~~~~~~~~~~~~~')
    cv2.waitKey(0)
    return (length, width, pixel)


# We start by cutting out the part of the picture where the thorax is

# This is the path of the picture for windows system
# path = str('C:\\Users\\vllaurens\\Desktop\\Spot_images\\SOKOL16\\So05D_16.jpg')

# This is the path of the picture for IOS system
# TODO: Manually type in the name of the image for now. Can be automated in further development.
name = "019A_R_22_SA.tif"
print("Processing image: {} \n".format(name))
path = os.path.expanduser("~/St.George/2022Summer/WorkStudy/Project/pics/" + name)

# Coordinators for landmarks
upper_x = '2200'
upper_y = '400'

lower_x = '2200'
lower_y = '800'


im = cv2.imread(path)

# Return the dimension of the given image as a tuple
# tuple[0]: height of image
# tuple[1]: width of image
# tuple[2]: number of channels (colors) in the image.
# eg. RGB image has 3 channels, Grayscale image has only 1 channel
height, width, channels = im.shape

# Creates a black image with the same size of the wing image
mask = np.zeros((height, width), np.uint8)

# The following part is processing the image to make sure only the 500 nm line and the spot are detected. #

# Coordination can be replaced by landmarks.
# TODO: Learn ImageJ to get landmarks!!
u_x = '1000'
u_y = '400'
l_x = '1000'

# l_y can be fixed to height - 200.
l_y = height - 200

pts2 = np.array([[u_x, 0], [l_x, l_y], [width, l_y], [width, 0]])

# Draw contours of the shape bounded by new landmarks
_ = cv2.drawContours(mask, np.int32([pts2]), 0, 255, -1)

# Uncomment the following to look at the mask with landmarks drawn
# cv2.imshow('landmarks2', mask)
# cv2.waitKey(0)


raw_wing2 = im.copy()

# Cut out the unwanted part
# "mask" is initialized as an array with only 0's, which indicates as "black" color.
# If there are any other colors exists, the int would be greater than 0 in the array.

raw_wing2[mask > 0] = 255  # cut area will be white (255)
cv2.line(raw_wing2, (int(u_x), 0), (int(l_x), int(l_y)),
         (255, 255, 255), thickness=8, lineType=cv2.LINE_AA)

# Uncomment the following to look at raw_wing2
# cv2.imshow('raw_wing2', raw_wing2)
# cv2.waitKey(0)


# Then the spot contour is estimated
blurred = cv2.pyrMeanShiftFiltering(raw_wing2, 31, 91)  # dilate & erode doesn't work well

gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

retval, threshold = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)

# Uncomment the following to see the threshold
# cv2.imshow('thresh_spot', threshold)
# cv2.waitKey(0)

cont_s, _ = cv2.findContours(threshold, cv2.RETR_LIST,
                             cv2.CHAIN_APPROX_NONE)

# The second-largest contour would be the wing spot
spot_ith_larg, spot_cont_ind = contour_index(cont_s, 1)
cv2.drawContours(im, cont_s, spot_cont_ind, (0, 255, 0), 6)

# Returns the number of non-zero pixels of this contour in the image.
# Green's theorem states that the line integral is equal to the
# double integral of this quantity over the enclosed region.
areaSpot = cv2.contourArea(spot_ith_larg)
print("The spot contour has {} pixels.\n".format(areaSpot))


# The third-largest contour would be the 500nm line
line_ith_larg, line_cont_ind = contour_index(cont_s, 2)
cv2.drawContours(im, cont_s, line_cont_ind, (0, 255, 0), 6)
areaLine = cv2.contourArea(line_ith_larg)
print("The scale bar contour has {} pixels.\n".format(areaLine))


# Uncomment the following to see the contours drawn
# cv2.imshow('Disp', im)
# cv2.waitKey(0)


line_length, line_width, line_pixel = compute_size(im, cont_s[line_cont_ind], None, "Scale bar")
spot_length, spot_width, spot_pixel = compute_size(im, cont_s[spot_cont_ind], line_pixel, "Spot")

# Writes the information to txt file
try:
    # Try open the file if it already exists
    f = open("results_for_testing.txt", 'a')
except FileNotFoundError:
    # If not exists, create a new file
    f = open("results_for_testing.txt", 'w')

f.write(str(name[:-4]) + ',' + '{:.2f}'.format(spot_length) + ',' + '{:.2f}'.format(spot_width) + '\n')
