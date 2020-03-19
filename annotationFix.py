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

def getContourCenter(contour):
    m = cv.moments(contour)
    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])
    cv.circle(result, (cx, cy), 7, (255, 255, 255), -1)
    cv.putText(result, "center", (cx - 20, cy - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return cx, cy

def getEdgeContour(coordinates, center, position):
    if position == "left":
        coordinates = sorted(coordinates, key=lambda x: x[0] + x[1])
        bottom_right = coordinates[-1]
        coordinates = sorted(coordinates, key=lambda x: x[0] + (3000 - x[1]))
        top_right = coordinates[-1]
        array = [center, bottom_right, bottom_red, (cX_red, cY_red), top_red, top_right]
    elif position == "right":
        coordinates = sorted(coordinates, key=lambda x: x[0] + x[1])
        top_left = coordinates[0]
        coordinates = sorted(coordinates, key=lambda x: (3000 - x[0]) + x[1])
        bottom_left = coordinates[-1]
        array = [center, bottom_left, bottom_red, (cX_red, cY_red), top_red, top_left]
    elif position == "around":
        array = [top_green_biggest, right_green_biggest, bottom_green_biggest, left_green_biggest]
    return array


# Load image
image = cv.imread('images/2part1.png')
imghsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
font = cv.FONT_HERSHEY_COMPLEX

# Create emtpy mask
mask = np.zeros(image.shape[:2], dtype=image.dtype)

# Find red contours
lower_red1 = np.array([0, 245, 245])
upper_red1 = np.array([0, 255, 255])
lower_red2 = np.array([0, 255, 255])
upper_red2 = np.array([0, 255, 255])
mask_red1 = cv.inRange(imghsv, lower_red1, upper_red1)
mask_red2 = cv.inRange(imghsv, lower_red2, upper_red2)
mask_red = mask_red1 + mask_red2
contours_red, _ = cv.findContours(mask_red, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
biggest_contour_red = max(contours_red, key=cv.contourArea)
# Create list of red contours to fill inside
contours_red_final = list()
contours_red_final.append(biggest_contour_red)
# Draw all red contours but the biggest one on the mask
for c in contours_red:
    if not np.array_equal(biggest_contour_red, c):
        cv.drawContours(mask, [c], 0, 255, -1)
# Get coordinates of the biggest contour
left_red = tuple(biggest_contour_red[biggest_contour_red[:, :, 0].argmin()][0])
right_red = tuple(biggest_contour_red[biggest_contour_red[:, :, 0].argmax()][0])
top_red = tuple(biggest_contour_red[biggest_contour_red[:, :, 1].argmin()][0])
bottom_red = tuple(biggest_contour_red[biggest_contour_red[:, :, 1].argmax()][0])


# Find green contours
lower_green = np.array([50, 50, 120])
upper_green = np.array([70, 255, 255])
mask_green = cv.inRange(imghsv, lower_green, upper_green)
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
left_green_biggest = tuple(biggest_contour_green[biggest_contour_green[:, :, 0].argmin()][0])
right_green_biggest = tuple(biggest_contour_green[biggest_contour_green[:, :, 0].argmax()][0])
# Draw all green contours but the biggest one on the mask
for c in contours_green:
    if not np.array_equal(biggest_contour_green, c):
        # Remove small contours
        if cv.contourArea(c) < 5000:
            cv.drawContours(mask, [c], 0, 255, -1)
            continue
        # Get coordinates of other contours
        coordinates_green = getContourCoordinates(c)
        coordinates_green = sorted(coordinates_green, key=lambda tup: tup[1])
        top_green_other = coordinates_green[0]
        bottom_green_other = coordinates_green[-1]
        # Check that coordinates match
        if bottom_green_other[1] < top_green_biggest[1] or top_green_other[1] > bottom_green_biggest[1]:
            cv.drawContours(mask, [c], 0, 255, -1)
        else:
            contours_green_final.append(c)

# Find blue contours
lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])
mask_blue = cv.inRange(imghsv, lower_blue, upper_blue)
contours_blue, _ = cv.findContours(mask_blue, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
biggest_contour_blue = max(contours_blue, key=cv.contourArea)
# Draw all blue contours but the biggest one on the mask
for c in contours_blue:
    if not np.array_equal(biggest_contour_blue, c):
        cv.drawContours(mask, [c], 0, 255, -1)
contours_blue_final = list()
contours_blue_final.append(biggest_contour_blue)

# Apply the mask to the original image
mask = cv.bitwise_not(mask)
result = cv.bitwise_and(image, image, mask=mask)

# Fill inside of final contours with correct color
cv.drawContours(result, contours_red_final, -1, (0, 0, 255), -1)
cv.drawContours(result, contours_green_final, -1, (0, 255, 0), -1)
cv.drawContours(result, contours_blue_final, -1, (255, 0, 0), -1)

# Compute the center of contours
# Red
cX_red, cY_red = getContourCenter(biggest_contour_red)
# Green
green_position = ""
if len(contours_green_final) == 1:
    cX_green, cY_green = getContourCenter(biggest_contour_green)
    if cX_green < left_red[0]:
        green_position = "left"
    elif cX_green > right_red[0]:
        green_position = "right"
    else:
        green_position = "around"
elif len(contours_green_final) == 2:
    cX_green, cY_green = getContourCenter(biggest_contour_green)
    cX_green_2, cY_green_2 = getContourCenter(contours_green_final[1])
    green_position = "two_part"
else:
    print("Error")

if green_position == "left":
    edge_contour = [np.array([getEdgeContour(coordinates_green_biggest, (cX_green, cY_green), "left")], dtype=np.int32)]
elif green_position == "right":
    edge_contour = [np.array([getEdgeContour(coordinates_green_biggest, (cX_green, cY_green), "right")], dtype=np.int32)]
elif green_position == "around":
    edge_contour = [np.array([getEdgeContour(coordinates_green_biggest, (cX_green, cY_green), "around")], dtype=np.int32)]
elif green_position == "two_part":
    if cX_green < cX_green_2:
        left_green_contour = coordinates_green_biggest
        left_green_center = (cX_green, cY_green)
        right_green_contour = coordinates_green
        right_green_center = (cX_green_2, cY_green_2)
    else:
        right_green_contour = coordinates_green_biggest
        right_green_center = (cX_green, cY_green)
        left_green_contour = coordinates_green
        left_green_center = (cX_green_2, cY_green_2)
    edge_contour = [np.array([getEdgeContour(left_green_contour, left_green_center, "left")], dtype=np.int32),
                    np.array([getEdgeContour(right_green_contour, right_green_center, "right")], dtype=np.int32)]
for cnt in edge_contour:
    # cv.drawContours(result, edge_contour, 0, (0, 255, 0), -1)
    cv.drawContours(result, cnt, 0, (255, 255, 255), 2)

# Fill inside of final contours with correct color
# cv.drawContours(result, contours_red_final, -1, (255, 0, 255), 3)
# cv.drawContours(result, contours_green_final, -1, (0, 255, 0), -1)
# cv.drawContours(result, contours_blue_final, -1, (255, 0, 0), -1)


# Show image
cv.imwrite("output.png", result)

cv.waitKey(0)
cv.destroyAllWindows()

