import cv2
import numpy as np
from math import sqrt
import sys


def correctWhiteBalance(img, area):
    # make a binary mask from the reference area
    mask = np.full(img.shape[:2], 0, dtype=np.uint8)
    cv2.fillPoly(
        mask,
        np.array([area], "int32"),
        255,
    )

    # imCopy = cv2.bitwise_and(img, img, mask=mask)
    # cv2.imwrite(f"{name}_part.jpg", imCopy)

    # get all the area's pixels
    height, width, _ = img.shape
    pixels = np.array(
        [img[y][x] for x in range(width) for y in range(height) if mask[y][x]]
    )

    # https://codeandlife.com/2019/08/17/correcting-image-white-balance-with-python-pil-and-numpy/
    c = list(np.mean(pixels[:, i]) for i in range(3))
    minc = float(min(c))

    wb = img.astype(float)
    for i in range(3):
        wb[:, :, i] /= c[i] / minc

    # cv2.imwrite(f"{name}_wb.jpg", wb)

    img = wb.astype(np.uint8)
    return img


def writeText(img, text):
    cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


def measure(img, area):
    x1, y1 = area[1]
    x3, y3 = area[3]

    catX = x3 - x1
    catY = y3 - y1
    hypoPix = sqrt(catX ** 2 + catY ** 2)
    pxRatio = hypoPix / 200  # pixels/mm

    height, width, _ = img.shape
    return int(width / pxRatio), int(height / pxRatio)


name = sys.argv[1]

filename = f"{name}.jpg"
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, corners = cv2.findChessboardCorners(gray, (3, 9), None)

area = [corners[0][0], corners[2][0], corners[26][0], corners[24][0]]

img = correctWhiteBalance(img, area)

size = measure(img, area)
width, height = size
writeText(img, f"FOV {width} x {height} mm")

cv2.imwrite(f"{name}_1.jpg", img)
