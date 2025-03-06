import streamlit as st
import pickle
import numpy as np


@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

st.header("Patient Data & Prediction")

age = st.slider("Age", 0, 100, 50)
gender = st.radio("Gender", ["Male", "Female"])
direct_bilirubin = st.slider("Direct Bilirubin", 0.0, 20.0, 10.0)
alkaline_phosphotase = st.slider("Alkaline Phosphotase", 0, 2000, 1000)
alamine_aminotransferase = st.slider("Alamine Aminotransferase", 0.0, 173.0, 100.0)
aspartate_aminotransferase = st.slider("Aspartate Aminotransferase", 0, 5000, 2500)
total_proteins = st.slider("Total Proteins", 0.0, 10.0, 5.0)
albumin = st.slider("Albumin", 0.0, 6.0, 3.0)
albumin_and_globulin_ratio = st.slider("Albumin and Globulin Ratio", 0.0, 3.0, 1.5)

gender_numeric = 1 if gender == "Male" else 0

input_data = np.array([
    age, gender_numeric, direct_bilirubin, alkaline_phosphotase,
    alamine_aminotransferase, aspartate_aminotransferase,
    total_proteins, albumin, albumin_and_globulin_ratio
]).reshape(1, -1)

st.write("### Collected Patient Data:")
st.json({
    "Age": age,
    "Gender": gender,
    "Direct Bilirubin": direct_bilirubin,
    "Alkaline Phosphotase": alkaline_phosphotase,
    "Alamine Aminotransferase": alamine_aminotransferase,
    "Aspartate Aminotransferase": aspartate_aminotransferase,
    "Total Proteins": total_proteins,
    "Albumin": albumin,
    "Albumin and Globulin Ratio": albumin_and_globulin_ratio
})


if st.button("Predict"):
    prediction = model.predict(input_data)[0]  

    if prediction == 1:
        st.error("⚠️ High Risk: The model predicts a potential liver disease.")
    else:
        st.success("✅ Low Risk: The model predicts a healthy liver condition.")
