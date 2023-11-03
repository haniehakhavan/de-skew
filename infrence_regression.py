import cv2
import numpy as np
import onnxruntime

# Load the ONNX model
model_path = "path/to/onnx/model"
onnx_model = onnxruntime.InferenceSession(model_path)
image_path = "path/to/image"


def preprocess_image(image_path):
    # Load the image and convert it to grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 127.5 - 1  # Apply the same normalization as during training
    image = image.reshape(1, 224, 224, 1)  # Add batch dimension
    return image


def regress_image(image_path):
    # Preprocess the image
    image = preprocess_image(image_path)

    # Perform inference using the ONNX model
    input_name = onnx_model.get_inputs()[0].name
    output_name = onnx_model.get_outputs()[0].name
    result = onnx_model.run([output_name], {input_name: image})

    # Extract the regression result
    regression_result = result[0]

    return regression_result


result = regress_image(image_path)
skew = result[0][0] * 45
print(skew)  # Interpret the output as needed
