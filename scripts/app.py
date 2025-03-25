import streamlit as st
import sqlite3
import tensorflow as tf
import numpy as np
import cv2
import os
import gdown  # For downloading from Google Drive
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define categories (same as training labels)
CATEGORIES = ["Hyperthyroid", "Hypothyroid", "Thyroid Cancer", "Thyroiditis", "Thyroid Nodules"]

# Google Drive File IDs (Replace these with your actual file IDs)
MODEL_FILE_ID = "https://drive.google.com/drive/folders/1TmDn4NmTdMs4SZNqj-6EfSKc5_Gp7zxM?usp=drive_link"
DATASET_FILE_ID = "https://drive.google.com/drive/folders/1eYTKfs7BaC7SlrOnWSGxwNIwka_rZAeO?usp=drive_link"

# Paths to store the downloaded files
MODEL_PATH = "models/thyroid_model.h5"
DATASET_PATH = "dataset/X.npy"

# Function to download files from Google Drive
@st.cache_data
def download_file(file_id, output_path):
    """Downloads a file from Google Drive and saves it locally."""
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)

# Download files if they don’t exist
if not os.path.exists(MODEL_PATH):
    st.write("🔽 Downloading model file from Google Drive...")
    download_file(MODEL_FILE_ID, MODEL_PATH)

if not os.path.exists(DATASET_PATH):
    st.write("🔽 Downloading dataset file from Google Drive...")
    download_file(DATASET_FILE_ID, DATASET_PATH)

# Load trained model for thyroid prediction
model = tf.keras.models.load_model(MODEL_PATH)

# Load chatbot model (Runs Offline)
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
chatbot_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Database connection function
def connect_db():
    return sqlite3.connect("users.db")

# Function to register a new user
def register_user(username, password):
    conn = connect_db()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return "Account successfully created"
    except sqlite3.IntegrityError:
        return "Username already exists"
    finally:
        conn.close()

# Function to login user
def login_user(username, password):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = cursor.fetchone()
    conn.close()
    return user is not None

# Function to predict uploaded image
def predict_image(image):
    img = Image.open(image)
    img = img.resize((224, 224))  # Resize to model input
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    return CATEGORIES[predicted_class], confidence * 100

# Function to get chatbot response
def chatbot_response(user_input):
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    response_ids = chatbot_model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

# Streamlit App
st.title("Thyroid Diagnosis System")

# Sidebar for Login/Register
menu = st.sidebar.selectbox("Menu", ["Login", "Register", "Chatbot"])

if menu == "Register":
    st.sidebar.subheader("Create New Account")
    new_user = st.sidebar.text_input("Username")
    new_pass = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Register"):
        st.sidebar.success(register_user(new_user, new_pass))

elif menu == "Login":
    st.sidebar.subheader("Login to Your Account")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if login_user(username, password):
            st.sidebar.success("Login Successful!")

            # Main App UI
            st.subheader("Upload an Image for Thyroid Diagnosis")
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)

                if st.button("Predict"):
                    result, confidence = predict_image(uploaded_file)
                    st.success(f"Prediction: {result}\nConfidence: {confidence:.2f}%")
        else:
            st.sidebar.error("Invalid username or password")

elif menu == "Chatbot":
    st.subheader("AI Chatbot")
    user_input = st.text_input("Ask a question:")
    if st.button("Get Response"):
        response = chatbot_response(user_input)
        st.write("Chatbot:", response)