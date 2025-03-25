import os
import tensorflow as tf
import numpy as np

MODEL_PATH = "C:/Users/Monika/.vscode/Thyroid_Diagnosis_Project/models/thyroid_model.h5"

# Step 1: Check if the model file exists
if os.path.exists(MODEL_PATH):
    print("✅ Model file found!")
else:
    print("❌ Model file NOT found. Check the path or retrain your model.")
    exit()

# Step 2: Try loading the model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Step 3: Check if the model can make a prediction
try:
    dummy_input = np.random.rand(1, 224, 224, 3)  # Random image-like input
    prediction = model.predict(dummy_input)
    print("✅ Model prediction successful!")
    print("Prediction output:", prediction)
except Exception as e:
    print(f"❌ Error during prediction: {e}")
