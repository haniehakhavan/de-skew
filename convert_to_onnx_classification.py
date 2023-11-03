import tf2onnx
import onnx
from tensorflow import keras

model_path = "path/to/your/keras/model/weight"

# your_model_architecture
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dense(4, activation='softmax'))  # 4 classes for orientations

model.load_weights(model_path)

onnx_model, _ = tf2onnx.convert.from_keras(model)

onnx.save_model(onnx_model, "classification.onnx")
