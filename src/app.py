
import streamlit as st
import pandas as pd
import joblib
from google import genai
from dotenv import load_dotenv
load_dotenv()
import os
model = joblib.load("models/diabetes_xgboost_model.pkl")
features = joblib.load("models/feature_list.pkl")

client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))

st.title("AI Diabetes Risk Prediction System")
st.write("Enter your health parameters to determine diabetes risk.")

# Only show main 5 features
BMI = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
Age = st.slider("Age Group (1(18-24yrs),...,13(80+yrs))", 1, 13)
HighBP = st.selectbox("High Blood Pressure", [0,1])
HighChol = st.selectbox("High Cholesterol", [0,1])
PhysActivity = st.selectbox("Physical Activity", [0,1])


if st.button("Predict Diabetes Risk"):
    # Set default values for all other features
    user_data = {
        "BMI": BMI,
        "Age": Age,
        "HighBP": HighBP,
        "HighChol": HighChol,
        "CholCheck": 1,
        "Smoker": 0,
        "Stroke": 0,
        "HeartDiseaseorAttack": 0,
        "PhysActivity": PhysActivity,
        "Fruits": 1,
        "Veggies": 1,
        "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1,
        "NoDocbcCost": 0,
        "GenHlth": 3,
        "MentHlth": 0,
        "PhysHlth": 0,
        "DiffWalk": 0,
        "Sex": 0,
        "Education": 3,
        "Income": 4
    }

    input_df = pd.DataFrame([user_data])
    input_df = input_df[features]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Results of Prediction")

    if prediction == 1:
        st.error(f"High Diabetes Risk ({probability*100:.1f}%)")
    else:
        st.success(f"Low Diabetes Risk ({probability*100:.1f}%)")

    prompt = f"""
    A patient has the following health data:

    BMI: {BMI}
    Age group: {Age}
    High Blood Pressure: {HighBP}
    High Cholesterol: {HighChol}
    Physical Activity: {PhysActivity}

    The ML model predicts a diabetes risk probability of {probability*100:.1f}%.

    Explain the diabetes risk and provide:
    1. Lifestyle improvements
    2. Preventive health advice
    3. Diet suggestions
    """

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    st.subheader("AI Health Recommendations")
    st.write(response.text)