Go to mine -> final.py to run the code.


Since Python couldn't detect the 500 nm reference line from the original image, the code draws a line with the same size to the original image, save the image, and then read the new image.

OpenCV is used to detect the contour, and the contour list is sourted from top to bottom. Manually adjust the for-loop to make sure that the first contour detected is the reference line. Press <Enter> to iterate through each contour, and look for the one that estimates the wing size.

The result is printed on the console.
