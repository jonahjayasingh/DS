import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load the trained model
MODEL_PATH = "check.h5"

# Check if model exists
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file '{MODEL_PATH}' not found. Please train the model first.")
    st.stop()

# Load the model
try:
    model = load_model(MODEL_PATH)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Class Labels (Modify these based on your dataset)
CLASS_LABELS = {0: "Fake", 1: "Real", 2: "Damaged", 3: "Unknown"}

# Function to predict currency class
def predict_currency(image):
    try:
        img = image.convert('RGB')  # Convert to 3 channels
        img = img.resize((224, 224))  # Resize to match model input
        img = img_to_array(img) / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        predictions = model.predict(img)
        predicted_class = np.argmax(predictions)  # Get class index
        confidence = np.max(predictions) * 100  # Confidence percentage

        return CLASS_LABELS[predicted_class], confidence
    except Exception as e:
        return f"❌ Error processing image: {e}", 0

# Streamlit UI
st.title("💵 Indian Currency Detection")
st.write("Upload an image to check if the currency is **Real, Fake, Damaged, or Unknown**.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict on button click
    if st.button("🔍 Check Currency"):
        prediction, confidence = predict_currency(image)
        st.write(f"### 🔹 Prediction: **{prediction}**")
        st.write(f"### 🎯 Confidence: **{confidence:.2f}%**")
