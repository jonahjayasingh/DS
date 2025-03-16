import streamlit as st
import tensorflow as tf
import numpy as np
import json
import cv2
from PIL import Image

# Load model and class labels
MODEL_PATH = "hand_sign_model.h5"
LABELS_PATH = "class_indices.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    class_labels = json.load(f)

st.title("Hand Sign Recognition")
st.text("Upload an image or use webcam for real-time hand sign detection.")

# Upload image option
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

def predict_hand_sign(img):
    img = img.resize((150, 150))  # Resize
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    key = list(class_labels.keys())[list(class_labels.values()).index(predicted_class)]
    print(key)
    return key

# Predict uploaded image
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    prediction = predict_hand_sign(image)
    st.write(f"Predicted Hand Sign: **{prediction}**")

# Webcam for real-time detection
st.text("Or use your webcam for live predictions.")

WEBCAM_ACTIVE = st.checkbox("Enable Webcam")

if WEBCAM_ACTIVE:
    cap = cv2.VideoCapture(0)
    stframe = st.image([])

    while WEBCAM_ACTIVE:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        prediction = predict_hand_sign(img_pil)

        cv2.putText(frame, f"Prediction: {prediction}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        stframe.image(frame, channels="BGR")

    cap.release()
