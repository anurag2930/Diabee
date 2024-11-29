import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests

# Set page config
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="wide")

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('diabetes_prediction_model.pkl')

model = load_model()

# Function to predict diabetes
def predict_diabetes(gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c, blood_glucose):
    # Encode categorical variables
    gender = 1 if gender == 'Female' else 0
    smoking_history = 1 if smoking_history == 'Yes' else 0

    # Prepare the input data as a DataFrame
    input_data = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'smoking_history': [smoking_history],
        'bmi': [bmi],
        'HbA1c_level': [hba1c],
        'blood_glucose_level': [blood_glucose]
    })

    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[0][1]
    return prediction[0], prediction_proba

# Function to load Lottie animations
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        r.raise_for_status()  # Raises a HTTPError if the status is 4xx, 5xx
        return r.json()
    except requests.exceptions.RequestException as e:
        st.warning(f"Error loading Lottie animation: {e}")
        return None

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border-radius: 5px;
    }
    .stTextInput>div>div>input {
        background-color: #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("About")
st.sidebar.info("This app predicts the likelihood of diabetes based on various health factors. Please consult with a healthcare professional for accurate medical advice.")

# Main content
st.title("🩺 Diabetes Prediction App")

# Load and display Lottie animation
lottie_url = "https://assets10.lottiefiles.com/packages/lf20_q5pk6p1k.json"
lottie_json = load_lottieurl(lottie_url)
if lottie_json:
    st_lottie(lottie_json, speed=1, height=200, key="initial")
else:
    st.image("https://via.placeholder.com/400x200.png?text=Diabetes+Prediction+App", use_column_width=True)

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.header("📋 Patient Information")
    gender = st.selectbox("Gender", options=["Male", "Female"])
    age = st.slider("Age", min_value=0, max_value=120, value=30)
    hypertension = st.checkbox("Hypertension")
    heart_disease = st.checkbox("Heart Disease")
    smoking_history = st.radio("Smoking History", options=["Yes", "No"])

with col2:
    st.header("📊 Health Metrics")
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    hba1c = st.number_input("HbA1c Level (%)", min_value=0.0, max_value=15.0, value=5.0, step=0.1)
    blood_glucose = st.number_input("Blood Glucose Level (mg/dL)", min_value=0.0, max_value=500.0, value=100.0, step=1.0)

    # BMI Category
    if bmi < 18.5:
        bmi_category = "Underweight"
    elif 18.5 <= bmi < 25:
        bmi_category = "Normal weight"
    elif 25 <= bmi < 30:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obese"
    
    st.info(f"BMI Category: {bmi_category}")

# Prediction button
if st.button("Predict"):
    prediction, prediction_proba = predict_diabetes(
        gender, age, int(hypertension), int(heart_disease), 
        smoking_history, bmi, hba1c, blood_glucose
    )
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction_proba * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Diabetes Risk", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': 'green'},
                {'range': [20, 40], 'color': 'lime'},
                {'range': [40, 60], 'color': 'yellow'},
                {'range': [60, 80], 'color': 'orange'},
                {'range': [80, 100], 'color': 'red'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))

    fig.update_layout(height=300)
    
    st.plotly_chart(fig, use_container_width=True)
    
    if prediction == 1:
        st.error("The model predicts that the patient **has a high risk of diabetes**.")
    else:
        st.success("The model predicts that the patient **has a low risk of diabetes**.")
    
    st.write(f"Probability of diabetes: {prediction_proba:.2%}")

# Display health tips
st.header("🌟 Health Tips")
tips = [
    "Maintain a balanced diet rich in fruits, vegetables, and whole grains.",
    "Exercise regularly, aiming for at least 150 minutes of moderate activity per week.",
    "Keep your weight in check and aim for a healthy BMI.",
    "Monitor your blood sugar levels regularly if you're at risk.",
    "Get enough sleep and manage stress through relaxation techniques.",
    "Limit alcohol consumption and avoid smoking."
]
for tip in tips:
    st.markdown(f"- {tip}")

# Disclaimer
st.caption("Disclaimer: This app is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.")