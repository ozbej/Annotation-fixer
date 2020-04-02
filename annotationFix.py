import numpy as np
import cv2 as cv


def getContourCoordinates(contour):
    coordinates = list()
    approx_biggest = cv.approxPolyDP(contour, 0.009 * cv.arcLength(contour, True), True)
    n = approx_biggest.ravel()
    i = 0
    for j in n:
        if i % 2 == 0:
            x = n[i]
            y = n[i + 1]
            coordinates.append((x, y))
        i += 1
    return coordinates


def processImage():
    # Load image
    imghsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Find red contours
    mask_red1 = cv.inRange(imghsv, np.array([0, 245, 245]), np.array([0, 255, 255]))
    mask_red2 = cv.inRange(imghsv, np.array([0, 255, 255]), np.array([0, 255, 255]))
    mask_red = mask_red1 + mask_red2
    contours_red, _ = cv.findContours(mask_red, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    biggest_contour_red = max(contours_red, key=cv.contourArea)

    # Find green contours
    mask_green = cv.inRange(imghsv, np.array([50, 50, 120]), np.array([70, 255, 255]))
    contours_green, _ = cv.findContours(mask_green, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    biggest_contour_green = max(contours_green, key=cv.contourArea)
    # Create list of contours to fill inside
    contours_green_final = list()
    contours_green_final.append(biggest_contour_green)
    # Get coordinates of the biggest green contour
    coordinates_green_biggest = getContourCoordinates(biggest_contour_green)
    coordinates_green_biggest = sorted(coordinates_green_biggest, key=lambda tup: tup[1])
    top_green_biggest = coordinates_green_biggest[0]
    bottom_green_biggest = coordinates_green_biggest[-1]
    # Draw all green contours but the biggest one on the mask
    for c in contours_green:
        if not np.array_equal(biggest_contour_green, c):
            # Remove small contours
            if cv.contourArea(c) < 5000:
                continue
            # Get coordinates of other contours
            coordinates_green = getContourCoordinates(c)
            coordinates_green = sorted(coordinates_green, key=lambda tup: tup[1])
            top_green_other = coordinates_green[0]
            bottom_green_other = coordinates_green[-1]
            coordinates_green = sorted(coordinates_green, key=lambda tup: tup[0])
            # Check that coordinates match
            print(bottom_green_other, top_green_biggest, top_green_other, bottom_green_biggest)
            if bottom_green_other[1] < top_green_biggest[1] or top_green_other[1] > bottom_green_biggest[1]:
                continue
            else:
                contours_green_final.append(c)

    # Find blue contours
    mask_blue = cv.inRange(imghsv, np.array([110, 50, 50]), np.array([130, 255, 255]))
    contours_blue, _ = cv.findContours(mask_blue, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    biggest_contour_blue = max(contours_blue, key=cv.contourArea)

    # Fill missing edges
    hull_red = cv.convexHull(biggest_contour_red, False)
    hull_green = []
    for cnt in contours_green_final:
        hull_green.append(cv.convexHull(cnt, False))
    cv.drawContours(original, hull_green, -1, (0, 255, 0), -1)
    cv.drawContours(original, [hull_red], -1, (0, 0, 255), -1)

    # Fill inside of final contours with correct color
    cv.drawContours(original, contours_green_final, -1, (0, 255, 0), -1)
    cv.drawContours(original, [biggest_contour_red], -1, (0, 0, 255), -1)
    cv.drawContours(original, [biggest_contour_blue], -1, (255, 0, 0), -1)

    # Write to original image
    cv.imwrite(output, original)


# Parametri
image = cv.imread('images/1_3i_Ru_1.png')
output = "output.png"
original = cv.imread('images/1_3i_Ru_1.jpg')

processImage()

