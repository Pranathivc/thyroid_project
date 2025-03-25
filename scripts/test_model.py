import numpy as np
import tensorflow as tf
import cv2
import os

# Load trained model
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(PROJECT_PATH, "models", "thyroid_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# Define categories (same as training labels)
CATEGORIES = ["Hyperthyroid", "Hypothyroid", "Thyroid Cancer", "Thyroiditis", "Thyroid Nodules"]

def predict_image(image_path):
    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize to match model input
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)  # Get the class index
    confidence = np.max(prediction)  # Get confidence score

    print(f"Predicted Condition: {CATEGORIES[predicted_class]} ({confidence*100:.2f}% confidence)")

# Test with a sample image
image_path = os.path.join(PROJECT_PATH, "dataset", "hyperthyroid", "sample.jpg")  # Change this to your test image
predict_image(image_path)