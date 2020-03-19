import numpy as np
import cv2 as cv

img = cv.imread('test2.png')
imghsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

#Red contours
lower_red1 = np.array([10, 245, 245])
upper_red1 = np.array([0, 255, 255])
lower_red2 = np.array([0, 255, 255])
upper_red2 = np.array([0, 255, 255])
mask_red1 = cv.inRange(imghsv, lower_red1, upper_red1)
mask_red2 = cv.inRange(imghsv, lower_red2, upper_red2)
mask_red = mask_red1 + mask_red2
contours_red, _ = cv.findContours(mask_red, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
biggest_contour_red = max(contours_red, key=cv.contourArea)
contours_red_final = list()
contours_red_final.append(biggest_contour_red)

#Green contours
lower_green = np.array([50, 50, 120])
upper_green = np.array([70, 255, 255])
mask_green = cv.inRange(imghsv, lower_green, upper_green)
contours_green, _ = cv.findContours(mask_green, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
biggest_contour_green = max(contours_green, key=cv.contourArea)

#Blue contours
lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])
mask_blue = cv.inRange(imghsv, lower_blue, upper_blue)
contours_blue, _ = cv.findContours(mask_blue, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
biggest_contour_blue = max(contours_blue, key=cv.contourArea)

im = np.copy(img)

#Draw RGB contours
cv.drawContours(im, contours_red_final, -1, (0, 0, 255), -1)
cv.drawContours(im, max(contours_green, key=cv.contourArea), -1, (0, 255, 0), -1)
cv.drawContours(im, max(contours_blue, key=cv.contourArea), -1, (255, 0, 0), -1)

print(type(contours_red), type(max(contours_red, key=cv.contourArea)), type(biggest_contour_red))
print(contours_red)

cv.imwrite("output.png", im)
