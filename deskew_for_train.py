import os
import math

import cv2
from deskew import determine_skew
from pytesseract import Output
import pytesseract

data_path = "path/to/data"
output_dir = "path/to/output/dir"


def rotate_image(image, angleInDegrees):
    h, w = image.shape[:2]
    img_c = (w / 2, h / 2)

    rot = cv2.getRotationMatrix2D(img_c, angleInDegrees, 1)

    rad = math.radians(angleInDegrees)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    outImg = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    return outImg


for im in os.listdir(data_path):
    im_path = os.path.join(data_path, im)
    image = cv2.imread(im_path)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)
    rotated = rotate_image(image, angle)
    results = pytesseract.image_to_osd(rotated, output_type=Output.DICT)
    new_image = rotate_image(image, -results["rotate"])
    out_path = os.path.join(output_dir, im)
    cv2.imwrite(out_path, rotated)
