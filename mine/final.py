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
    try:
        ith_largest_contour = sorteddata[ith_bigger][1]
    except IndexError:
        # When the scale bar is overlapping with the fly body, Python couldn't detect its contour.
        print("!! Failed to detect contour !!")
        return (None, -1)
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

# This is the template of the folder path for windows system
# path = str('C:\\Users\\vllaurens\\Desktop\\Spot_images\\SOKOL16\\So05D_16.jpg')

# This is the template of the picture for IOS system
# path = os.path.expanduser("~/St.George/2022Summer/WorkStudy/Project/")

# TODO: Only works for OS system. Need to fix this issue.

def find_spot_area(path, folder, name):
    s = "~~~ Processing image: {} ~~~".format(name)
    print(s)
    image = path + '/' + folder + '/' + name
    im = cv2.imread(image)
    height, width, channels = im.shape
    mask = np.zeros((height, width), np.uint8)
    u_x = '1000'
    u_y = '400'
    l_x = '1000'
    l_y = height - 200
    pts2 = np.array([[u_x, 0], [l_x, l_y], [width, l_y], [width, 0]])
    _ = cv2.drawContours(mask, np.int32([pts2]), 0, 255, -1)
    raw_wing2 = im.copy()
    raw_wing2[mask > 0] = 255  # cut area will be white (255)
    cv2.line(raw_wing2, (int(u_x), 0), (int(l_x), int(l_y)),
             (255, 255, 255), thickness=8, lineType=cv2.LINE_AA)
    blurred = cv2.pyrMeanShiftFiltering(raw_wing2, 31, 91)  # dilate & erode doesn't work well
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    retval, threshold = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    cont_s, _ = cv2.findContours(threshold, cv2.RETR_LIST,
                                 cv2.CHAIN_APPROX_NONE)

    spot_ith_larg, spot_cont_ind = contour_index(cont_s, 1)
    if spot_cont_ind >= 0:
        cv2.drawContours(im, cont_s, spot_cont_ind, (0, 255, 0), 6)
        areaSpot = cv2.contourArea(spot_ith_larg)
        print("The spot contour has {} pixels.".format(areaSpot))
    else:
        areaSpot = -1

    line_ith_larg, line_cont_ind = contour_index(cont_s, 2)
    if line_cont_ind >= 0:
        cv2.drawContours(im, cont_s, line_cont_ind, (0, 255, 0), 6)
        areaLine = cv2.contourArea(line_ith_larg)
        print("The scale bar contour has {} pixels.".format(areaLine))
    else:
        areaLine = -1

    # Writes the information to txt file
    if os.path.exists("for_testing.txt"):
        # Try open the file if it already exists
        f = open("for_testing.txt", 'a')
    else:
        # If not exists, create a new file
        f = open("for_testing.txt", 'w')
        f.write("Folder,Image_Name,Spot_Pixel,ScaleBar_Pixel" + '\n')

    f.write(str(folder + ',' + name[:-4]) + ',' + '{}'.format(areaSpot) + ',' + '{}'.format(areaLine) + '\n')
    f.close()
    print("~" * len(s) + '\n')
    return





# path = os.path.expanduser("~/St.George/2022Summer/WorkStudy/Project/")
# dir = "pics"
# name = "022A_R_22_SA.tif"
# print("Processing image: {} \n".format(name))
#
# find_spot_area(path, dir, name)


# Path of the directory which contains subdirectories of images.
path = os.path.expanduser("~/St.George/2022Summer/WorkStudy/Project/samples/")

# Get a list of each image folder
dirlist = [ item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item)) ]

for direct in dirlist:
    directory = path + '/' + direct
    print("*** Folder: {} ***".format(direct))
    for filename in os.listdir(directory):
        if filename.endswith(".tif"):
            find_spot_area(path, direct, filename)
            continue
        else:
            continue
    print("*** {} Completed *** \n".format(direct))
