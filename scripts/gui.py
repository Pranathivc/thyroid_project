import tkinter as tk
from tkinter import messagebox, filedialog, scrolledtext
import sqlite3
import cv2
import numpy as np
import tensorflow as tf
import os
from PIL import Image, ImageTk
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load trained model for thyroid prediction
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(PROJECT_PATH, "models", "thyroid_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# Define categories (same as training labels)
CATEGORIES = ["Hyperthyroid", "Hypothyroid", "Thyroid Cancer", "Thyroiditis", "Thyroid Nodules"]

# Load chatbot model (Runs Offline)
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
chatbot_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Database connection function
def connect_db():
    return sqlite3.connect("users.db")

# Function to register a new user
def register_user():
    username = entry_username.get()
    password = entry_password.get()

    if not username or not password:
        messagebox.showerror("Error", "Please enter username and password")
        return

    conn = connect_db()
    cursor = conn.cursor()
    
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        messagebox.showinfo("Success", "Account successfully created")
    except sqlite3.IntegrityError:
        messagebox.showerror("Error", "Username already exists")
    
    conn.close()

# Function to login user
def login_user():
    username = entry_username.get()
    password = entry_password.get()

    conn = connect_db()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = cursor.fetchone()
    
    conn.close()

    if user:
        messagebox.showinfo("Success", "Login successful")
        login_window.destroy()
        open_main_gui()
    else:
        messagebox.showerror("Error", "Invalid username or password")

# Function to predict uploaded image
def predict_image():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    
    if not file_path:
        return  # If no file selected, return
    
    # Load and display the image
    img = Image.open(file_path)
    img = img.resize((250, 250))  # Resize for display
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk  # Keep reference to prevent garbage collection

    # Preprocess image for model prediction
    img = cv2.imread(file_path)
    img = cv2.resize(img, (224, 224))  # Resize to match model input
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)  # Get the class index
    confidence = np.max(prediction)  # Get confidence score

    # Update the result label with the prediction
    result_label.config(
        text=f"Predicted Condition:\n{CATEGORIES[predicted_class]}\nConfidence: {confidence*100:.2f}%",
        fg="white", bg="#444", font=("Arial", 14, "bold")
    )

# Function to open chatbot window
def open_chatbot():
    chat_window = tk.Toplevel()
    chat_window.title("AI Chatbot")
    chat_window.geometry("500x500")
    chat_window.config(bg="#222")

    # Chat display area
    chat_display = scrolledtext.ScrolledText(chat_window, wrap=tk.WORD, width=60, height=20, bg="black", fg="white", font=("Arial", 12))
    chat_display.pack(pady=10)
    chat_display.insert(tk.END, "Chatbot: Hello! Ask me anything.\n")

    # Entry field for user input
    entry_chat = tk.Entry(chat_window, font=("Arial", 12), width=50)
    entry_chat.pack(pady=5)

    # Function to send message
    def send_message():
        user_input = entry_chat.get()
        if not user_input:
            return
        chat_display.insert(tk.END, f"You: {user_input}\n")
        entry_chat.delete(0, tk.END)

        # Generate chatbot response
        input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
        response_ids = chatbot_model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        bot_response = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

        # Display chatbot response
        chat_display.insert(tk.END, f"Chatbot: {bot_response}\n")

    # Send button
    send_button = tk.Button(chat_window, text="Send", command=send_message, font=("Arial", 12), bg="green", fg="white")
    send_button.pack(pady=10)

# Function to open the main GUI after login
def open_main_gui():
    global image_label, result_label

    root = tk.Tk()
    root.title("Thyroid Diagnosis System")
    root.geometry("500x700")
    root.config(bg="#222")

    # Heading Label
    heading = tk.Label(root, text="Thyroid Diagnosis System", font=("Arial", 16, "bold"), fg="white", bg="#222")
    heading.pack(pady=10)

    # Image Label (For displaying uploaded image)
    image_label = tk.Label(root, bg="#222")
    image_label.pack(pady=10)

    # Upload Button
    upload_button = tk.Button(root, text="Upload Image", command=predict_image, padx=20, pady=10, font=("Arial", 12),
                              bg="#007BFF", fg="white", relief="raised")
    upload_button.pack(pady=10)

    # Prediction Result Label
    result_label = tk.Label(root, text="", font=("Arial", 14), fg="white", bg="#222")
    result_label.pack(pady=10)

    # Chatbot Button
    chatbot_button = tk.Button(root, text="Chat with AI", command=open_chatbot, padx=20, pady=10, font=("Arial", 12),
                               bg="#FF4500", fg="white", relief="raised")
    chatbot_button.pack(pady=20)

    root.mainloop()

# Create Login Window
login_window = tk.Tk()
login_window.title("User Login")
login_window.geometry("400x300")
login_window.config(bg="#222")

# Login Labels and Entries
tk.Label(login_window, text="Username:", font=("Arial", 12), fg="white", bg="#222").pack(pady=5)
entry_username = tk.Entry(login_window, font=("Arial", 12))
entry_username.pack(pady=5)

tk.Label(login_window, text="Password:", font=("Arial", 12), fg="white", bg="#222").pack(pady=5)
entry_password = tk.Entry(login_window, show="*", font=("Arial", 12))
entry_password.pack(pady=5)

# Buttons
login_button = tk.Button(login_window, text="Login", command=login_user, padx=20, pady=5, font=("Arial", 12), bg="green", fg="white")
login_button.pack(pady=10)

register_button = tk.Button(login_window, text="Register", command=register_user, padx=20, pady=5, font=("Arial", 12), bg="blue", fg="white")
register_button.pack(pady=10)

login_window.mainloop()