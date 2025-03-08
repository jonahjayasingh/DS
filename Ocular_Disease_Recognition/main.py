import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load trained models
numerical_model = pickle.load(open("model.pkl", "rb"))  # Logistic Regression Model
image_model = tf.keras.models.load_model("model.h5")  # CNN Model

# Define labels
labels = {
    0: "Normal (N)",
    1: "Diabetes (D)",
    2: "Glaucoma (G)",
    3: "Cataract (C)",
    4: "Age-related Macular Degeneration (A)",
    5: "Hypertension (H)",
    6: "Pathological Myopia (M)",
    7: "Other diseases/abnormalities (O)"
}

# Function to preprocess image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Function to predict disease from numerical input
def predict_numerical(features):
    prediction = numerical_model.predict([features])
    return labels[prediction[0]]

# Function to predict disease from image
def predict_image(img_path):
    img_array = preprocess_image(img_path)
    prediction = image_model.predict(img_array)
    predicted_label = np.argmax(prediction[0])
    return labels[predicted_label]

# Streamlit UI
st.title("Disease Prediction System")
st.sidebar.header("Choose Input Method")

option = st.sidebar.radio("Select Input Type", ("Numerical Data", "Image Upload"))

if option == "Numerical Data":
    st.header("Enter Patient Details")

    # Input fields
    age = st.number_input("Patient Age", min_value=1, max_value=120, value=50)
    sex = st.radio("Patient Sex", ("Male", "Female"))
    sex = 0 if sex == "Male" else 1

    st.subheader("Select Diagnostic Keywords")
    left_diag = st.selectbox("Left Eye Diagnostic", ["Cataract", "Glaucoma", "Normal Fundus", "Pathological Myopia"])
    right_diag = st.selectbox("Right Eye Diagnostic", ["Cataract", "Glaucoma", "Normal Fundus", "Pathological Myopia"])

    # Mapping diagnostic keywords to numbers
    diag_mapping = {
        "Cataract": 0,
        "Glaucoma": 1,
        "Normal Fundus": 2,
        "Pathological Myopia": 3
    }
    left_diag = diag_mapping[left_diag]
    right_diag = diag_mapping[right_diag]

    # Additional disease flags (can be modified as needed)
    diseases = ["N", "D", "G", "C", "A", "H", "M", "O"]
    disease_values = [st.checkbox(f"Has {d}?") for d in diseases]
    disease_values = [1 if d else 0 for d in disease_values]

    # Predict button
    if st.button("Predict Disease"):
        features = [age, sex, left_diag, right_diag] + disease_values
        result = predict_numerical(features)
        st.success(f"Predicted Disease: **{result}**")

elif option == "Image Upload":
    st.header("Upload Retinal Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        # Save image temporarily
        img_path = "temp_image.jpg"
        with open(img_path, "wb") as f:
            f.write(uploaded_file.read())

        # Predict button
        if st.button("Predict Disease"):
            result = predict_image(img_path)
            st.success(f"Predicted Disease: **{result}**")
