import os
import math
import random

import cv2

# Set your source and output directories
source_directory = '/path/to/raw/images/dir'
output_directory = 'path/to/output/dir'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)


# Define the rotate_image function
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


# Process each image in the source directory
for filename in os.listdir(source_directory):
    image_path = os.path.join(source_directory, filename)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_name = os.path.splitext(filename)[0]

    # Randomly select 15 rotation angles
    # You can change k value to generate more data
    random_angles = random.sample(range(-45, 45), 15)

    for angle in random_angles:
        rotated_image = rotate_image(image.copy(), angle)

        angle_str = f"{angle:.1f}"
        output_filename = os.path.join(output_directory, f'{image_name}_{angle_str}.jpg')
        cv2.imwrite(output_filename, rotated_image)
