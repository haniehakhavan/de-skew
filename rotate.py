import os

import cv2
import numpy as np

dataset_dir = "data/path"
output_dir = "output/dir"
rotation_value = -0.1  # choose value that you want to rotate your images


def rotate_image(image, angle):
    # Calculate the image center
    image_center = tuple(np.array(image.shape[1::-1]) / 2)

    # Define the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    return rotated_image


for im in os.listdir(dataset_dir):
    image_path = os.path.join(dataset_dir, im)
    image = cv2.imread(image_path)
    rotated = rotate_image(image, rotation_value)
    output_path = os.path.join(output_dir, im)
    cv2.imwrite(output_path, rotated)
