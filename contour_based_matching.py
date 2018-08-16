import cv2
import numpy as np

x_coordinate = []
y_coordinate = []

# Image read and convert to Gray Scale

img1 = cv2.imread('Instruments/6.jpg')
img2 = cv2.imread('Instruments/18.jpg')
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

kernel_big = np.ones((23, 23), np.uint8)

# Pass images through Gaussian Blur and morphological operations

img1_gray = cv2.GaussianBlur(img1_gray, (5, 5), 0)
img1_gray = cv2.morphologyEx(img1_gray, cv2.MORPH_OPEN, kernel_big)
img1_gray = cv2.morphologyEx(img1_gray, cv2.MORPH_CLOSE, kernel_big)

kernel = np.ones((11, 11), np.uint8)
img2_gray = cv2.GaussianBlur(img2_gray, (5, 5), 0)
img2_gray = cv2.morphologyEx(img2_gray, cv2.MORPH_OPEN, kernel)
img2_gray = cv2.morphologyEx(img2_gray, cv2.MORPH_CLOSE, kernel)

# Find contours in the images

ret, thresh = cv2.threshold(img1_gray, 127, 255, 0)
ret, thresh2 = cv2.threshold(img2_gray, 127, 255, 0)
img1_c, contours, hierarchy = cv2.findContours(thresh, 2, 1)
cnt1 = contours
print(len(contours))
img2_c, contours, hierarchy = cv2.findContours(thresh2, 2, 1)
print(len(contours))
cnt2 = contours[0]

# Show contours

for contour in contours:
    # get rectangle bounding contour
    [x,y,w,h] = cv2.boundingRect(cnt2)

    # draw rectangle around contour on original image
    cv2.rectangle(img2,(x, y),(x+w, y+h), (0, 0, 255), 4)

# Find matching contours in both the images

for contour in cnt1:
    # get rectangle bounding contour
    [x, y, w, h] = cv2.boundingRect(contour)
    x_coordinate.append(0.5*(x+w))
    y_coordinate.append(0.5*(y+h))
    # draw rectangle around contour on original image
    cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 0, 255), 4)


for i, m in enumerate(cnt1):
    ret = cv2.matchShapes(m, cnt2, 1, 0.0)
    n = cv2.moments(m)
    print(ret, x_coordinate[i], y_coordinate[i])

img1 = cv2.resize(img1, (960, 960))
img2 = cv2.resize(img2, (960, 960))

# Show contours

cv2.imshow('Image1 contour', img1)
cv2.imshow('Image2 contour', img2)

cv2.waitKey(0)
