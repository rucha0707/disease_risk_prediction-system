
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
st.write("Enter your health parameters to determine diabetes risk.(0-No,1-Yes)")

# Only show main 5 features which were calculated in the notebook
BMI = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
Age = st.slider("Age Groups (1(18-24yrs),2(25-29yrs),3(30-34yrs),4(35-39yrs),5(40-44yrs),6(45-49yrs),7(50-54yrs),8(55-59yrs),9(60-64yrs),10(65-69yrs),11(70-74yrs),12(75-79yrs),13(80+yrs))", 1, 13)
HighBP = st.selectbox("High Blood Pressure", [0,1])
HighChol = st.selectbox("High Cholesterol", [0,1])
PhysActivity = st.selectbox("Physical Activity", [0,1])


if st.button("Predict Diabetes Risk"):
    #setting defalt values for the other features which we are not consideribg for the input ui part
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
#health recommendations based on the predction of the risk
    st.subheader("Health Recommendations")

    if prediction == 1:
        st.markdown("""
### High Diabetes Risk Detected

Based on your health inputs, you may be at **higher risk for diabetes**.  
Consider the following preventive steps:

**Lifestyle Changes**
- Maintain a healthy body weight
- Exercise at least **30 minutes daily**
- Reduce sugar and refined carbohydrate intake
- Increase fiber-rich foods (vegetables, whole grains)

**Diet Recommendations**
- Choose whole grains instead of white rice or refined flour
- Avoid sugary drinks and processed snacks
- Eat more fruits, vegetables, and lean proteins

**Health Monitoring**
- Get **regular blood glucose testing**
- Monitor blood pressure and cholesterol levels
- Consult a healthcare professional for proper diagnosis

### Useful Resources
- https://www.cdc.gov/diabetes/prevention
- https://www.who.int/news-room/fact-sheets/detail/diabetes
- https://www.niddk.nih.gov/health-information/diabetes
""")
    else:
        st.markdown("""
### Low Diabetes Risk

Your current inputs indicate a **lower risk of diabetes**, but maintaining healthy habits is important.

**Healthy Lifestyle Tips**
- Stay physically active (at least **150 minutes per week**)
- Maintain a balanced diet
- Limit processed and high-sugar foods
- Maintain a healthy weight

**Preventive Measures**
- Get regular health checkups
- Monitor blood sugar levels if you have family history
- Manage stress and sleep well

### Helpful Articles
- https://www.cdc.gov/diabetes/prevention
- https://www.healthline.com/nutrition/prevent-diabetes
- https://diabetes.org/about-diabetes/diabetes-prevention
""")