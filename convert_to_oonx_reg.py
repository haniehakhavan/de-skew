import tf2onnx
import onnx
import tensorflow as tf

model_path = "path/to/your/keras/model/weight"

# your_model_architecture
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 1)),  # Input shape for grayscale images
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    tf.keras.layers.GlobalMaxPooling2D(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation=tf.keras.activations.tanh)
])

model.load_weights(model_path)

onnx_model, _ = tf2onnx.convert.from_keras(model)

onnx.save_model(onnx_model, "regresion.onnx")

