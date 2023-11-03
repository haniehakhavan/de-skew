import cv2
import os
import random
import math

# Set your source and output directories
source_directory = '/path/to/raw/images/dir'
output_directory = 'path/to/output/dir'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

orientations = ["up", "down", "left", "right"]
for orientation in orientations:
    dir_path = os.path.join(source_directory, orientation)
    os.makedirs(dir_path, exist_ok=True)


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
    image = cv2.imread(image_path)
    image_name = os.path.splitext(filename)[0]

    random_angles = random.sample(range(-10, 10), 3)
    for angle in random_angles:
        rotated_image = rotate_image(image.copy(), angle)
        angle_str = f"{angle:.1f}"
        output_filename = os.path.join(output_directory, "up", f'{image_name}_{angle_str}_up.jpg')
        cv2.imwrite(output_filename, rotated_image)

    random_angles = random.sample(range(-10, 10), 3)
    for angle in random_angles:
        angle = angle + 180
        rotated_image = rotate_image(image.copy(), angle)
        angle_str = f"{angle:.1f}"
        output_filename = os.path.join(output_directory, "down", f'{image_name}_{angle_str}_down.jpg')
        cv2.imwrite(output_filename, rotated_image)

    random_angles = random.sample(range(-10, 10), 3)
    for angle in random_angles:
        angle = angle + 90
        rotated_image = rotate_image(image.copy(), angle)
        angle_str = f"{angle:.1f}"
        output_filename = os.path.join(output_directory, "left", f'{image_name}_{angle_str}_left.jpg')
        cv2.imwrite(output_filename, rotated_image)

    random_angles = random.sample(range(-10, 10), 3)
    for angle in random_angles:
        angle = angle + 270
        rotated_image = rotate_image(image.copy(), angle)
        angle_str = f"{angle:.1f}"
        output_filename = os.path.join(output_directory, "right", f'{image_name}_{angle_str}_right.jpg')
        cv2.imwrite(output_filename, rotated_image)
