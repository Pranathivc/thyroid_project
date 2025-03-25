import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import os

# Get dataset path
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(PROJECT_PATH, "dataset")

# Load preprocessed data
X = np.load(os.path.join(DATASET_PATH, "X.npy"))
Y = np.load(os.path.join(DATASET_PATH, "Y.npy"))

# Split data into training (80%) and testing (20%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Build CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')  # 5 classes (Hypothyroidism, Hyperthyroidism, etc.)
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=32)

# Save the trained model
MODEL_PATH = os.path.join(PROJECT_PATH, "models")
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

model.save(os.path.join(MODEL_PATH, "thyroid_model.h5"))

print(" Model training complete. Model saved successfully!")