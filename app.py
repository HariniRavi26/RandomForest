import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor


st.title(
"💰 Salary Prediction using Random Forest"
)

st.write(
"Predict employee salary using experience, education, skills and projects"
)


#################################

# LOAD DATA

#################################

df = pd.read_csv(

"salary_data.csv"

)


#################################

# FEATURES

#################################

X = df.drop(

'Salary',

axis=1

)

y = df['Salary']


#################################

# MODEL

#################################

model = RandomForestRegressor(

n_estimators=100,

random_state=42

)

model.fit(

X,

y

)


#################################

# INPUTS

#################################

st.header(
"Enter Employee Details"
)


experience = st.number_input(

"Experience",

1,

20,

5

)


education = st.number_input(

"Education",

12,

18,

14

)


age = st.number_input(

"Age",

20,

50,

25

)


skills = st.slider(

"Skills Score",

50,

100,

70

)


certifications = st.number_input(

"Certifications",

0,

10,

2

)


hours = st.number_input(

"Working Hours",

35,

50,

40

)


projects = st.number_input(

"Projects Completed",

1,

20,

5

)


#################################

# PREDICT

#################################

if st.button(

"Predict Salary"

):


    input_data = pd.DataFrame({

'Experience':[experience],

'Education':[education],

'Age':[age],

'Skills_Score':[skills],

'Certifications':[certifications],

'Working_Hours':[hours],

'Projects':[projects]

})


    prediction = model.predict(

        input_data

    )


    st.success(

f"Predicted Salary: ₹ {prediction[0]:,.0f}"

)

