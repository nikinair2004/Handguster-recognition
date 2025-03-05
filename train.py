import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import cv2

# Define constants
img_size = 300
labels = ["down", "fist", "okay", "palm", "thumbs", "up"]

# Data paths
data_dir = "Data"  # Replace with your path
categories = labels

# Prepare Data
def load_data():
    data = []
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = labels.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                img_resized = cv2.resize(img_array, (img_size, img_size))
                data.append([img_resized, class_num])
            except Exception as e:
                pass
    return data

data = load_data()

# Shuffle the data
import random
random.shuffle(data)

# Split features and labels
X = []
y = []

for features, label in data:
    X.append(features)
    y.append(label)

# Convert lists to numpy arrays and normalize pixel values
X = np.array(X).reshape(-1, img_size, img_size, 1) / 255.0
y = np.array(y)

# One-hot encode the labels
y = to_categorical(y, num_classes=len(labels))
#One-hot encoding: Converts the numeric labels (e.g., 0, 1, etc.) into a binary matrix.
# For example, if you have 6 gesture types, a label might look like [1, 0, 0, 0, 0, 0].

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Data Augmentation
train_datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# Build CNN Model
model = Sequential()

# Layer 1 - Convolution + Pooling
model.add(Conv2D(32, (3, 3), input_shape=(img_size, img_size, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 2 - Convolution + Pooling
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 3 - Convolution + Pooling
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Fully connected layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dropout for regularization
model.add(Dense(len(labels), activation='softmax'))  # Output layer for classification

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))

# Save the model
model.save("Model3/keras_model.h5")
