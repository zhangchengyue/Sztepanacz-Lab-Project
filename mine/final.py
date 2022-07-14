import os
import cv2
import imutils
import numpy as np
from imutils import perspective
from imutils import contours
from scipy.spatial import distance as dist

# Global variable: Path of the original image
path = os.path.expanduser("~/St.George/2022Summer/WorkStudy/Project/pics/0018A_R_22_SA.tif")


def draw_and_save(new_path, image):
    """
    A function that draws a line in the image, save it, and then re-read the image for further process.
    The line can be used as a size reference of 500 nm in length.

    :param new_path: The new path where the new image is saved.
    :param image: The original image that needs to draw a line on.
    :return: New image with a line.
    """
    cv2.line(image, (204, 144), (712, 144), (0, 0, 0), thickness=12)
    cv2.imwrite(new_path, image)
    result = cv2.imread(new_path)
    return result


def contour_index(contours, ith_bigger):
    """Finds the ith biggest contour among a set of contours.

    Args:
        contours: a set of contours as obtained from cv2.findContours
        ith_bigger: 0 for the biggest, 1 for the 2nd biggest, etc

    Returns:
        The ith biggest contour and its index in the original set of contours
    """
    areaArray = []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        areaArray.append(area)
    sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)
    ith_largest_contour = sorteddata[ith_bigger][1]
    cont_index = areaArray.index(cv2.contourArea(ith_largest_contour))
    return (ith_largest_contour, cont_index)


def midpoint(ptA, ptB):
    """
    Detect the midpoint of two coordinators.
    Returns the coordinator of the midpoint.
    """
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


image = cv2.imread(path)
im = draw_and_save(os.path.expanduser("~/St.George/2022Summer/WorkStudy/Project/line/018_line.tif"), image)

######################################
# # Increase the contrast of the image
# lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
# l_channel, a, b = cv2.split(lab)
#
# # Applying CLAHE to L-channel
# # feel free to try different values for the limit and grid size:
# clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(6, 6))
# cl = clahe.apply(l_channel)
# limg = cv2.merge((cl, a, b))
# enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
# im = enhanced_img
######################################


# We detect the wing contour by setting up a colour threshold

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

dark = cv2.equalizeHist(gray)

ret, thresh = cv2.threshold(dark, 90, 255, cv2.THRESH_BINARY_INV)

cont, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

ith_largest, cont_index = contour_index(cont, 0)

# Whole wing
cv2.drawContours(im, cont, cont_index, (255, 0, 0), 6)

cv2.namedWindow('Disp', cv2.WINDOW_NORMAL)

areaWing = cv2.contourArea(ith_largest)

# Then the spot contour is estimated

blurred = cv2.pyrMeanShiftFiltering(im, 31, 30)  # dilate & erode doesn't work well

gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

retval, threshold = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)

cnts, hierarchies = cv2.findContours(threshold, cv2.RETR_LIST,
                                     cv2.CHAIN_APPROX_NONE)

spot_ith_larg, spot_cont_ind = contour_index(cnts, 1)

# Right part of the wing
cv2.drawContours(im, cnts, spot_cont_ind, (255, 0, 0), 6)

(cnts, _) = contours.sort_contours(cnts, 'top-to-bottom')
pixelsPerMetric = None


# Loop over the contours individually
for c in cnts[5:]:
    # If the contour is not sufficiently large or too large, ignore it
    if cv2.contourArea(c) < 7000 or cv2.contourArea(c) > 2000000:
        continue
    # Compute the rotated bounding box of the contour
    orig = im.copy()
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    # Order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding
    # box
    box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
    # loop over the original points and draw them
    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

    # Unpack the ordered bounding box, then compute the midpoint
    # between the top-left and top-right coordinates, followed by
    # the midpoint between bottom-left and bottom-right coordinates
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    # Compute the midpoint between the top-left and top-right points,
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
    if pixelsPerMetric is None:
        pixelsPerMetric = dB / 500

    # compute the size of the object
    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric
    # draw the object sizes on the image
    cv2.putText(orig, "{:.1f} nm".format(dimA),
                (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    cv2.putText(orig, "{:.1f} nm".format(dimB),
                (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    # show the output image
    cv2.imshow("Image", orig)
    # print('Contour size is: {}'.format(cv2.contourArea(c)))
    # print('Coordinator is: {}'.format(c))
    print('Object size :\n' + 'Length: {:.1f} nm \n'.format(dimB) + 'Width: {:.1f} nm \n'.format(dimA) + '~~~~~~~~~~~~~~')
    cv2.waitKey(0)

