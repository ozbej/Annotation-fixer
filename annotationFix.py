import numpy as np
import cv2 as cv

# load image
image = cv.imread('images/right1.png')
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
coordinates_green_biggest = list()
approx_biggest = cv.approxPolyDP(biggest_contour_green, 0.009 * cv.arcLength(biggest_contour_green, True), True)
n = approx_biggest.ravel()
i = 0
for j in n:
    if i % 2 == 0:
        x = n[i]
        y = n[i+1]
        coordinates_green_biggest.append((x, y))
    i += 1
coordinates_green_biggest = sorted(coordinates_green_biggest, key=lambda tup: tup[1])
top_green_biggest = coordinates_green_biggest[0]
bottom_green_biggest = coordinates_green_biggest[-1]
# Draw all green contours but the biggest one on the mask
for c in contours_green:
    if not np.array_equal(biggest_contour_green, c):
        # Remove small contours
        if cv.contourArea(c) < 5000:
            cv.drawContours(mask, [c], 0, 255, -1)
            continue
        # Get coordinates of other contours
        coordinates_green = list()
        approx = cv.approxPolyDP(c, 0.009 * cv.arcLength(c, True), True)
        n = approx.ravel()
        i = 0
        for j in n:
            if i % 2 == 0:
                x = n[i]
                y = n[i + 1]
                coordinates_green.append((x, y))
            i += 1
        coordinates_green = sorted(coordinates_green, key=lambda tup: tup[1])
        top_green_other = coordinates_green[0]
        bottom_green_other = coordinates_green[-1]
        # Check that coordinates match and that contour is big enough
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
M_red = cv.moments(biggest_contour_red)
cX_red = int(M_red["m10"] / M_red["m00"])
cY_red = int(M_red["m01"] / M_red["m00"])
cv.circle(result, (cX_red, cY_red), 7, (255, 255, 255), -1)
cv.putText(result, "center RED", (cX_red - 20, cY_red - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Green
green_position = ""
if len(contours_green_final) == 1:
    M_green = cv.moments(biggest_contour_green)
    cX_green = int(M_green["m10"] / M_green["m00"])
    cY_green = int(M_green["m01"] / M_green["m00"])
    cv.circle(result, (cX_green, cY_green), 7, (255, 255, 255), -1)
    cv.putText(result, "center GREEN", (cX_green - 20, cY_green - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    if cX_green < left_red[0]:
        print("Beločnica je levo")
        green_position = "left"
    elif cX_green > right_red[0]:
        print("Beločnica je desno")
        green_position = "right"
    else:
        print("Beločnica je okoli")
        green_position = "around"
elif len(contours_green_final) == 2:
    for c in contours_green_final:
        M_green = cv.moments(c)
        cX_green = int(M_green["m10"] / M_green["m00"])
        cY_green = int(M_green["m01"] / M_green["m00"])
        cv.circle(result, (cX_green, cY_green), 7, (255, 255, 255), -1)
        cv.putText(result, "center GREEN", (cX_green - 20, cY_green - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    print("Beločnica je dvo delna")
    green_position = "two_part"
else:
    print("Error")

if green_position == "left":
    coordinates_green_biggest = sorted(coordinates_green_biggest, key=lambda x: x[0] + x[1])
    bottom_right = coordinates_green_biggest[-1]
    coordinates_green_biggest = sorted(coordinates_green_biggest, key=lambda x: x[0] + (3000 - x[1]))
    top_right = coordinates_green_biggest[-1]
    cv.circle(result, bottom_right, 7, (255, 255, 255), -1)
    cv.circle(result, top_right, 7, (255, 255, 255), -1)
    array = [(cX_green, cY_green), bottom_right, bottom_red, (cX_red, cY_red), top_red, top_right]
    edge_contour = [np.array([array], dtype=np.int32)]
    #cv.drawContours(result, edge_contour, 0, (0, 255, 0), -1)
    cv.drawContours(result, edge_contour, 0, (255, 255, 255), 2)
elif green_position == "right":
    coordinates_green_biggest = sorted(coordinates_green_biggest, key=lambda x: x[0] + x[1])
    top_left = coordinates_green_biggest[0]
    coordinates_green_biggest = sorted(coordinates_green_biggest, key=lambda x: (3000 - x[0]) + x[1])
    bottom_left = coordinates_green_biggest[-1]
    cv.circle(result, top_left, 7, (0, 255, 255), -1)
    cv.circle(result, bottom_left, 7, (255, 255, 255), -1)
    array = [(cX_green, cY_green), bottom_left, bottom_red, (cX_red, cY_red), top_red, top_left]
    edge_contour = [np.array([array], dtype=np.int32)]
    # cv.drawContours(result, edge_contour, 0, (0, 255, 0), -1)
    cv.drawContours(result, edge_contour, 0, (255, 255, 255), 2)

cv.circle(result, top_red, 7, (255, 255, 255), -1)
cv.circle(result, bottom_red, 7, (255, 255, 255), -1)
cv.circle(result, left_red, 7, (255, 255, 255), -1)
cv.circle(result, right_red, 7, (255, 255, 255), -1)

# Fill inside of final contours with correct color
cv.drawContours(result, contours_red, -1, (255, 255, 255), 2)
#cv.drawContours(result, contours_green_final, -1, (0, 255, 0), -1)
#cv.drawContours(result, contours_blue_final, -1, (255, 0, 0), -1)


# Show image
cv.imwrite("output.png", result)

cv.waitKey(0)
cv.destroyAllWindows()

