# de-skew scanned documents



Welcome to the Document Skew Detection project! This repository contains a collection of code and tools designed to address the task of detecting and correcting skew in scanned documents. The project is divided into multiple components, each contributing to the overall goal of training a model to accurately determine the orientation and skew angle of scanned documents.


<br/>


<br/>

![python](https://img.shields.io/static/v1?label=python&message=v3.8&color=FCA7D5)

<h3>Table of Contents</h3>

- [Requirement](#requirement)
- [Project Structure](#Project Structure)
- [Project Objective](#Project Objective)
- [Getting Started](#Getting Started)

## requierments

Python 3.8
#### command:
```sh
pip install -r requirements.txt
```

## Project Structure

The project is organized into several key components, each with its specific functionality:

1. **Deskew Data (Code: `deskew_for_train.py`)**:
   This script is responsible for deskewing input images and saving the corrected images to a designated directory. Properly aligned images are crucial for accurate model training and testing.

2. **Rotate Images (Code: `rotate.py`)**:
   This script allows you to rotate images by specifying the desired angle. It can be used to create augmented data for training and testing.

3. **Generate Data for Regression (Code: `create_regression_data.py`)**:
   Use this script to generate data suitable for regression problems. It prepares the dataset necessary for training the orientation detection model.

4. **Generate Data for Classification (Code: `create_classification_data.py`)**:
   This script is used to create data for classification tasks. It generates a dataset tailored for training the skew angle detection model.

5. **Train Classification Model (Code: `train_classification.py`)**:
   Train a machine learning or deep learning model to classify the skew angles of documents.

6. **Train Regression Model (Code: `train_regression_tfdata.py`)**:
   Train a model to predict the orientation (rotation) of scanned documents.

7. **Convert Keras Model to ONNX (Code: `convert_to_oonx_reg.py` and `create_classification_data.py`)**:
   These scripts facilitate the conversion of trained Keras models to ONNX format for deployment and use in different applications.

8. **Use Regression ONNX Model (Code: `infrence_regression.py`)**:
   This script demonstrates how to use the ONNX model for regression to predict document orientation.

9. **Use Classification ONNX Model (Code: `infrence_classification.py`)**:
   Use the ONNX model for classification to predict the skew angle of scanned documents.
   
## Project Objective

The main objective of this project is to train a model that can detect the skew of scanned documents. This task is divided into two key components:

1. **Orientation Detection**: Detect the orientation or rotation of the document.
2. **Skew Angle Detection**: Determine the skew angle of the document.

By combining these two components, you can accurately assess and correct skew in scanned documents.

## Getting Started

Before you start using this project, make sure you have all the necessary prerequisites and dependencies installed. Detailed instructions can be found in the respective script files and the project's documentation.

1. **Tesseract Installation**: You'll need to install Tesseract OCR. Follow the [Tesseract Installation Guide](https://tesseract-ocr.github.io/tessdoc/Installation.html) for detailed instructions on how to install Tesseract on your system.

