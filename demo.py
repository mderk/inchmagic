import cv2
import numpy as np
from math import sqrt
import sys


def getChessboardCorners(gray, rows, columns):
    ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return corners


def getChessboardAreaPoly(corners, rows, columns):
    length = rows * columns
    return [
        corners[0][0],
        corners[rows - 1][0],
        corners[length - 1][0],
        corners[length - rows][0],
    ]


def correctWhiteBalance(img, area):
    pixels = getReferenceAreaPixels(img, area)

    # return wb_rgb(img, pixels)
    return wb_ycbr(img, pixels)


def getReferenceAreaPixels(img, area):
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
    height, width = img.shape[:2]
    return np.array(
        [img[y][x] for x in range(width) for y in range(height) if mask[y][x]]
    )


# https://codeandlife.com/2019/08/17/correcting-image-white-balance-with-python-pil-and-numpy/
def wb_rgb(img, pixels):
    c = list(np.mean(pixels[:, i]) for i in range(3))
    minc = float(min(c))

    wb = img.astype(float)
    for i in range(3):
        wb[:, :, i] /= c[i] / minc

    # cv2.imwrite(f"{name}_wb.jpg", wb)
    return wb.astype(np.uint8)


def wb_ycbr(img, pixels):
    # Convert data and sample to YCbCr
    ycbcr = rgb2ycbcr(img)
    ysub = rgb2ycbcr(np.array([pixels]))

    yc = list(np.mean(ysub[:, :, i]) for i in range(3))

    for i in range(1, 3):
        ycbcr[:, :, i] = np.clip(ycbcr[:, :, i] + (128 - yc[i]), 0, 255)

    rgb = ycbcr2rgb(ycbcr)
    return rgb


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
    x0, y0 = area[0]
    x1, y1 = area[1]
    x2, y2 = area[2]
    x3, y3 = area[3]

    catX = x3 - x1
    catY = y3 - y1
    hypoPix = sqrt(catX ** 2 + catY ** 2)
    pxRatio = hypoPix / 200  # pixels/mm

    height, width = img.shape[:2]
    return int(width / pxRatio), int(height / pxRatio)


name = sys.argv[1]
filename = f"{name}.jpg"
img = cv2.imread(filename)

if img is not None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, columns = 3, 9
    corners = getChessboardCorners(gray, rows, columns)
    if corners is not None:
        area = getChessboardAreaPoly(corners, rows, columns)
        img = correctWhiteBalance(img, area)
        size = measure(img, area)
        width, height = size
        writeText(img, f"FOV {width} x {height} mm")
        cv2.drawChessboardCorners(img, (rows, columns), corners, True)
        cv2.imwrite(f"{name}_1.jpg", img)
