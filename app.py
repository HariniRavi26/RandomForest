import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


##########################################

# TITLE

##########################################

st.title("💰 Employee Salary Prediction App")

st.write(
    "Predict employee salary using Random Forest Regression"
)


##########################################

# LOAD DATA

##########################################

df = pd.read_csv(
    "salary_data.csv"
)


##########################################

# ENCODE CATEGORICAL DATA

##########################################

encoders = {}

categorical_columns = [

    'Education_Level',

    'Job_Role',

    'Company_Size',

    'Location'

]


for col in categorical_columns:

    le = LabelEncoder()

    df[col] = le.fit_transform(

        df[col]

    )

    encoders[col] = le


##########################################

# FEATURES + TARGET

##########################################

X = df.drop(

    'Salary',

    axis=1

)

y = df['Salary']


##########################################

# RANDOM FOREST MODEL

##########################################

model = RandomForestRegressor(

    n_estimators=100,

    random_state=42

)


model.fit(

    X,

    y

)


##########################################

# USER INPUTS

##########################################

st.header(

    "Enter Employee Details"

)


experience = st.number_input(

    "Experience",

    1,

    30,

    5

)


age = st.number_input(

    "Age",

    20,

    60,

    25

)


education = st.selectbox(

    "Education Level",

    encoders[
        'Education_Level'
    ].classes_

)


skills = st.slider(

    "Skills Score",

    50,

    100,

    75

)


certifications = st.number_input(

    "Certifications",

    0,

    20,

    2

)


projects = st.number_input(

    "Projects Completed",

    1,

    30,

    5

)


hours = st.number_input(

    "Working Hours",

    35,

    60,

    40

)


role = st.selectbox(

    "Job Role",

    encoders[
        'Job_Role'
    ].classes_

)


company = st.selectbox(

    "Company Size",

    encoders[
        'Company_Size'
    ].classes_

)


location = st.selectbox(

    "Location",

    encoders[
        'Location'
    ].classes_

)


performance = st.slider(

    "Performance Rating",

    1,

    5,

    4

)


promotions = st.number_input(

    "Promotions",

    0,

    5,

    0

)


##########################################

# PREDICTION

##########################################

if st.button(

    "Predict Salary"

):


    education_encoded = encoders[
        'Education_Level'
    ].transform(

        [education]

    )[0]


    role_encoded = encoders[
        'Job_Role'
    ].transform(

        [role]

    )[0]


    company_encoded = encoders[
        'Company_Size'
    ].transform(

        [company]

    )[0]


    location_encoded = encoders[
        'Location'
    ].transform(

        [location]

    )[0]



    input_data = pd.DataFrame({

        'Experience':[experience],

        'Age':[age],

        'Education_Level':[education_encoded],

        'Skills_Score':[skills],

        'Certifications':[certifications],

        'Projects_Completed':[projects],

        'Working_Hours':[hours],

        'Job_Role':[role_encoded],

        'Company_Size':[company_encoded],

        'Location':[location_encoded],

        'Performance_Rating':[performance],

        'Promotions':[promotions]

    })


    prediction = model.predict(

        input_data

    )


    st.success(

        f"Predicted Salary: ₹ {prediction[0]:,.0f}"

    )


