import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to dataset
DATASET_PATH = "dataset"  # Adjust path if needed
CATEGORIES = ["hyperthyroid", "hypothyroid", "thyroid_cancer", "thyroiditis", "thyroid_nodules"]

# Image settings
IMG_SIZE = 224  # Resize all images to 224x224
data = []
labels = []

# Load and preprocess images
for category in CATEGORIES:
    path = os.path.join(DATASET_PATH, category)
    label = CATEGORIES.index(category)  # Assign a number to each class
    
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize image
            img = img / 255.0  # Normalize pixel values
            data.append(img)
            labels.append(label)

# Convert to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Save preprocessed data
np.save(os.path.join(DATASET_PATH,"X.npy"),data)
np.save(os.path.join(DATASET_PATH,"Y.npy"),labels)

print("Preprocessing Complete. Data saved successfully.")