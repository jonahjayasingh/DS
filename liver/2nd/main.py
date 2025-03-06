import streamlit as st
import pickle
import numpy as np
import json


@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

st.title("Lung Cancer Survival Prediction")

age = st.slider("Age", 0, 100, 50)
gender = st.radio("Gender", ["Male", "Female"])
cancer_stage = st.radio("Cancer Stage", ["Stage 1", "Stage 2", "Stage 3", "Stage 4"])
family_history = st.radio("Family History", ["No", "Yes"])
smoking_status = st.radio("Smoking Status", ['Never Smoked','Passive Smoker', 'Former Smoker','Current Smoker'])
cholesterol = st.slider("Cholesterol Level", 100, 300, 200)
hypertension = st.radio("Hypertension", ["No", "Yes"])
asthma = st.radio("Asthma", ["No", "Yes"])
cirrhosis = st.radio("Cirrhosis", ["No", "Yes"])
other_cancer = st.radio("Other Cancer", ["No", "Yes"])
treatment_type = st.radio("Treatment Type", ['Combined','Radiation','Surgery','Chemotherapy'])

gender = 1 if gender == "Male" else 0


cancer_stage = int(cancer_stage.split(" ")[1])
family_history = 1 if family_history == "Yes" else 0
smoking_status = ['Never Smoked','Passive Smoker', 'Former Smoker','Current Smoker'].index(smoking_status)
hypertension = 1 if hypertension == "Yes" else 0
asthma = 1 if asthma == "Yes" else 0
cirrhosis = 1 if cirrhosis == "Yes" else 0
other_cancer = 1 if other_cancer == "Yes" else 0
treatment_type = ['Combined','Radiation','Surgery','Chemotherapy'].index(treatment_type)

patient_data = {
    "Age": age,
    "Gender": "Male" if gender else "Female",
    "Cancer Stage": f"Stage {cancer_stage}",
    "Family History": "Yes" if family_history else "No",
    "Smoking Status": ['Never Smoked','Passive Smoker', 'Former Smoker','Current Smoker'][smoking_status],
    "Cholesterol Level": cholesterol,
    "Hypertension": "Yes" if hypertension else "No",
    "Asthma": "Yes" if asthma else "No",
    "Cirrhosis": "Yes" if cirrhosis else "No",
    "Other Cancer": "Yes" if other_cancer else "No",
    "Treatment Type": ['Combined','Radiation','Surgery','Chemotherapy'][treatment_type]
}

st.subheader("Patient Data (JSON Format)")
st.json(patient_data)

input_data = np.array([
    age, gender, cancer_stage, family_history, 
    smoking_status, cholesterol, hypertension, asthma, 
    cirrhosis, other_cancer, treatment_type
]).reshape(1, -1)


if model and st.button("Predict"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Likely to survive")
    else:
        st.error("⚠️ Low survival probability. Consult a doctor.")
