import streamlit as st
import pickle
import numpy as np

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

st.header("Heart Disease Prediction")

age = st.slider("Age", 0, 100, 50)
gender = st.radio("Gender", ["Male", "Female"])
chest_pain_type = st.slider("Chest Pain Type", 0, 3, 2)
resting_blood_pressure = st.slider("Resting Blood Pressure", 90, 200, 120)
serum_cholestoral = st.slider("Serum Cholestoral", 120, 570, 250)
fasting_blood_sugar = st.slider("Fasting Blood Sugar", 0, 1, 0)
resting_electrocardiographic = st.slider("Resting Electrocardiographic", 0, 2, 1)
maximum_heart_rate_achieved = st.slider("Maximum Heart Rate Achieved", 70, 210, 150)
exercise_induced_angina = st.slider("Exercise Induced Angina", 0, 1, 1)
oldpeak = st.slider("Oldpeak", 0.0, 7.0, 3.0)
peak_exercise_st_segment = st.slider("Peak Exercise ST Segment", 0, 2, 1)
major_vessels_fluoroscopy = st.slider("Major Vessels (0-3) Colored by Fluoroscopy", 0, 3, 2)
thal = st.slider("thal", 0, 2, 1)

gender_numeric = 1 if gender == "Male" else 0

input_data = np.array([
    age, gender_numeric, chest_pain_type, resting_blood_pressure,
    serum_cholestoral, fasting_blood_sugar, resting_electrocardiographic,
    maximum_heart_rate_achieved, exercise_induced_angina, oldpeak,
    peak_exercise_st_segment, major_vessels_fluoroscopy,thal
]).reshape(1, -1)

st.write("### Collected Patient Data:")
st.json({
    "Age": age,
    "Gender": gender,
    "Chest Pain Type": chest_pain_type,
    "Resting Blood Pressure": resting_blood_pressure,
    "Serum Cholestoral": serum_cholestoral,
    "Fasting Blood Sugar": fasting_blood_sugar,
    "Resting Electrocardiographic": resting_electrocardiographic,
    "Maximum Heart Rate Achieved": maximum_heart_rate_achieved,
    "Exercise Induced Angina": exercise_induced_angina,
    "Oldpeak": oldpeak,
    "Peak Exercise ST Segment": peak_exercise_st_segment,
    "Major Vessels Fluoroscopy": major_vessels_fluoroscopy,
    "thal":thal
})

if st.button("Predict"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ High Risk: The model predicts a potential heart disease.")
    else:
        st.success("✅ Low Risk: The model predicts a healthy heart condition.")
