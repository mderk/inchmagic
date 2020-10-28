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

    # Convert data and sample to YCbCr
    ycbcr = rgb2ycbcr(img)
    ysub = rgb2ycbcr(np.array([pixels]))

    yc = list(np.mean(ysub[:, :, i]) for i in range(3))

    for i in range(1, 3):
        ycbcr[:, :, i] = np.clip(ycbcr[:, :, i] + (128 - yc[i]), 0, 255)

    rgb = ycbcr2rgb(ycbcr)
    return rgb

    # img = wb.astype(np.uint8)


# Conversion functions courtesy of https://stackoverflow.com/a/34913974/2721685
def rgb2ycbcr(im):
    xform = np.array(
        [[0.299, 0.587, 0.114], [-0.1687, -0.3313, 0.5], [0.5, -0.4187, -0.0813]]
    )
    ycbcr = im.dot(xform.T)
    ycbcr[:, :, [1, 2]] += 128
    return ycbcr  # np.uint8(ycbcr)


def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -0.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:, :, [1, 2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)


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
