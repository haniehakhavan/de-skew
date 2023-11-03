import os

import tensorflow as tf

data_dir = '/path/to/dataset/*.jpg'  # Replace with your directory path

batch_size = 32
img_height, img_width = 224, 224
buffer_size = 1024


def get_label(file_path):
    # Convert the path to a list of path components
    file_name = tf.strings.split(file_path, os.path.sep)[-1]
    # Get label from filename
    label = tf.strings.split(file_name, '_')[-1]
    label = tf.strings.split(label, '.')[0]
    label = tf.strings.to_number(label, out_type=tf.float32)
    return label


def load_image(file_path):
    # Load the image from the file
    image = tf.io.read_file(file_path)
    # Decode the image to a tensor
    image = tf.image.decode_jpeg(image, channels=1)
    return image


def process_path(file_path):
    label = get_label(file_path)
    image = load_image(file_path)
    image = tf.image.resize(image, [img_height, img_width])
    label = label / 45
    image = (image / 127.5) - 1
    return image, label


dataset = tf.data.Dataset.list_files(data_dir)
dataset = dataset.shuffle(buffer_size)
dataset = dataset.map(process_path)
dataset = dataset.batch(batch_size)
# Use prefetch to improve performance
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

total_size = dataset.cardinality().numpy()
train_size = int(0.8 * total_size)  # 80% of the data will be used for training

# Split the dataset into training and validation sets
train_dataset = dataset.take(train_size)
validation_dataset = dataset.skip(train_size)

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
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation=tf.keras.activations.tanh)
])

model.compile(optimizer='adam', loss="mean_squared_error")
model.fit(train_dataset, epochs=15, validation_data=validation_dataset)

val_loss = model.evaluate(validation_dataset)
print(f'Validation Loss: {val_loss}')
model.save("modelreg.h5")
model.save('modelreg.keras')
