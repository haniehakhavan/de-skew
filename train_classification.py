import tensorflow as tf
from tensorflow import keras

# Define your dataset directory and batch size
data_dir = "/path/to/dataset"
batch_size = 32  # You can adjust the batch size

# Define your CNN model
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

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Use image_dataset_from_directory with a validation split
train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='int',
    batch_size=batch_size,
    image_size=(224, 224),
    color_mode='grayscale',
    shuffle=True,
    seed=42,
    validation_split=0.2,  # Set the validation split (e.g., 20% for validation)
    subset='training'  # Choose 'training' subset
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='int',
    batch_size=batch_size,
    image_size=(224, 224),
    color_mode='grayscale',
    shuffle=True,
    seed=42,
    validation_split=0.2,  # Set the validation split (e.g., 20% for validation)
    subset='validation'  # Choose 'validation' subset
)

# Normalize the data between -1 and 1
train_dataset = train_dataset.map(lambda x, y: (x / 127.5 - 1, y))
validation_dataset = validation_dataset.map(lambda x, y: (x / 127.5 - 1, y))

# Prefetch data for better performance
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Train the model
history = model.fit(train_dataset, epochs=10, validation_data=validation_dataset)

# Evaluate the model
test_loss, test_acc = model.evaluate(validation_dataset)
print(f"Test accuracy: {test_acc}")

# Save the model (provide a path to save the model)
model.save('model.h5')
model.save('model.keras')


