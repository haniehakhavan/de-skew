import cv2
import numpy as np
import onnxruntime

# Load the ONNX model
model_path = "path/to/onnx/model"
onnx_model = onnxruntime.InferenceSession(model_path)
image_path = "path/to/image"


def preprocess_image(image):
    # Load the image and convert it to grayscale

    image = image.astype(np.float32) / 127.5 - 1  # Normalize pixel values to the range [-1, 1]
    image = image.reshape(1, 224, 224, 1)  # Add batch dimension
    return image


def classify_image(image):
    # Preprocess the image
    # Perform inference using the ONNX model
    input_name = onnx_model.get_inputs()[0].name
    output_name = onnx_model.get_outputs()[0].name
    result = onnx_model.run([output_name], {input_name: image})

    # Extract the classification result
    classification_result = result[0][0]

    return classification_result


image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (224, 224))
processed_image = preprocess_image(image)
pred = classify_image(processed_image)
predicted_orientation = np.argmax(pred)

class_labels = ["down", "left", "right", "up"]
predicted_class = class_labels[predicted_orientation]
orientation_degree_dict = {"down": 180,
                           "left": -90,
                           "right": 90,
                           "up": 0}
degree = orientation_degree_dict[predicted_class]

print(degree)
print(predicted_class)
