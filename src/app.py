import streamlit as st
import pandas as pd
import joblib
import google.generativeai as genai
import os

model=joblib.load(r'C:\Users\ksawa\Desktop\Hackathon\disease_risk_prediction-system\models\diabetes_xgboost_model.pkl')
features=joblib.load(r'C:\Users\ksawa\Desktop\Hackathon\disease_risk_prediction-system\models\feature_list.pkl')


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model=genai.GenerativeModel('gemini-1.5-flash')

print(os.getenv("GEMINI_API_KEY"))

#app title
st.title("AI Diabetes Risk Prediction System")

st.write("Enter your health parameters to determine diabetes risk.")

#user inputs

BMI=st.number_input("BMI",min_value=10.0,max_value=50.0,value=25.0,step=0.1)

Age = st.slider("Age Group (1-13)", 1, 13)

HighBP = st.selectbox("High Blood Pressure", [0,1])

HighChol = st.selectbox("High Cholesterol", [0,1])

CholCheck = st.selectbox("Cholesterol Check in Last 5 Years", [0,1])

Smoker = st.selectbox("Smoker", [0,1])

Stroke = st.selectbox("History of Stroke", [0,1])

HeartDiseaseorAttack = st.selectbox("Heart Disease or Attack", [0,1])

PhysActivity = st.selectbox("Physical Activity", [0,1])

Fruits = st.selectbox("Consumes Fruits Regularly", [0,1])

Veggies = st.selectbox("Consumes Vegetables Regularly", [0,1])

HvyAlcoholConsump = st.selectbox("Heavy Alcohol Consumption", [0,1])

AnyHealthcare = st.selectbox("Has Healthcare Access", [0,1])

NoDocbcCost = st.selectbox("Couldn't See Doctor Due to Cost", [0,1])

GenHlth = st.slider("General Health (1=Excellent,5=Poor)",1,5)

MentHlth = st.slider("Poor Mental Health Days (last 30 days)",0,30)

PhysHlth = st.slider("Poor Physical Health Days (last 30 days)",0,30)

DiffWalk = st.selectbox("Difficulty Walking", [0,1])

Sex = st.selectbox("Sex (0=Female,1=Male)", [0,1])

Education = st.slider("Education Level (1-6)",1,6)

Income = st.slider("Income Level (1-8)",1,8)

#prediction buttons
if st.button("Predict Diabetes Risk"):

    user_data = {
        "BMI":BMI,
        "Age":Age,
        "HighBP":HighBP,
        "HighChol":HighChol,
        "CholCheck":CholCheck,
        "Smoker":Smoker,
        "Stroke":Stroke,
        "HeartDiseaseorAttack":HeartDiseaseorAttack,
        "PhysActivity":PhysActivity,
        "Fruits":Fruits,
        "Veggies":Veggies,
        "HvyAlcoholConsump":HvyAlcoholConsump,
        "AnyHealthcare":AnyHealthcare,
        "NoDocbcCost":NoDocbcCost,
        "GenHlth":GenHlth,
        "MentHlth":MentHlth,
        "PhysHlth":PhysHlth,
        "DiffWalk":DiffWalk,
        "Sex":Sex,
        "Education":Education,
        "Income":Income
    }

    input_df=pd.DataFrame([user_data])

    input_df=input_df[features]

    #prediction by the model

    prediction=model.predict(input_df)[0]
    probability=model.predict_proba(input_df)[0][1]