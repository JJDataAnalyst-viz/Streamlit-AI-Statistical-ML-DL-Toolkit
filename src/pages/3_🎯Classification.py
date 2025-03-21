# standard Libs

import random
from components.utils.utils import avatar_func,rain_func
# Libraries for connection with DB
import mysql.connector
from dotenv import load_dotenv
import os


#Viz
import seaborn as sns
import matplotlib.pyplot as plt

# App
import streamlit as st


# DataFrame
import pandas as pd

# Scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier

#Xgboost
from xgboost import XGBClassifier


st.set_page_config(page_title='Classification problem',page_icon='🎯')

tabs1,tabs2,tabs3 = st.tabs(['Description','Data','Prediction'])
with tabs1:
    st.title("Road Accident Survival Dataset")

    st.write("""
    ### Overview:
    The **Road Accident Survival Dataset** is a simulated dataset designed to analyze the impact of key factors such as age, speed, and safety measures on the survival probability during road accidents. The data provides a comprehensive view of various demographic, behavioral, and situational attributes that contribute to the outcomes of road accidents.

    ### Key Features:
    - **Age**: The age of the individual involved in the accident. This feature helps analyze how age correlates with survival rates during a road accident.
    - **Speed**: The speed at which the vehicle was traveling at the time of the accident. It provides insights into how high-speed collisions affect the likelihood of survival.
    - **Safety Measures**: Whether safety measures, such as seatbelts and airbags, were used during the accident. This feature is crucial to understanding the role of safety measures in improving survival chances.
    - **Demographic Information**: Includes attributes like gender, and possibly other personal details, providing an opportunity to study how different demographics influence survival outcomes.
    - **Behavioral Data**: Contains data related to the individual’s behavior during the accident (e.g., seatbelt use, alcohol consumption, distraction levels, etc.).
    - **Accident Type**: The nature of the accident (e.g., frontal crash, side-impact, rollover), which helps correlate the severity and type of crash with survival chances.
    - **Outcome**: The survival status of the individual (survived or not), which is the target variable of interest for predictive analysis.

    ### Objectives:
    - **Analyze Survival Factors**: Investigate how age, speed, and safety measures impact survival rates.
    """)
    rain_func()
    avatar_func()
    

load_dotenv()
    
conn = mysql.connector.connect(
host = os.getenv("HOST_car"),
username = os.getenv("USERNAME_car"),
password = os.getenv("PASSWORD_car"),
port = os.getenv("PORT_car"),
database = os.getenv("DATABASE_car")
)

cursor = conn.cursor()

cursor.execute("SELECT * FROM car_acc")

res = cursor.fetchall()

columns = ['Age', 'Gender', 'Speed_of_Impact', 'Helmet_Used', 'Seatbelt_Used','Survived']

df = pd.DataFrame(res, columns=columns)


with tabs2:
    st.dataframe(df)



# --------------------- *******Make classification models********* ---------------------------- #


    dummy = pd.get_dummies(df.loc[:,['Gender','Helmet_Used','Seatbelt_Used']],dtype=int)
    dummy = dummy.loc[:,['Gender_Female','Helmet_Used_No','Seatbelt_Used_No']]
    dff = pd.concat([df.iloc[:,[0,2,5]],dummy],axis=1)




    X = dff.drop(columns='Survived')
    y = dff['Survived']

    y_streamlit = pd.DataFrame(data=df['Survived']).set_index(df['Gender'])
with tabs3:
    with st.expander('Choosen column for classification'):
        col1,col2 = st.columns(2)


        with col1:
            st.write(X)
        with col2:
            st.write(y_streamlit)

    st.session_state.X = X
    st.session_state.y = y


    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    st.write(X_train.shape)
    st.write(X_test.shape)
    st.write(y_train.shape)
    st.write(y_test.shape)
    st.write(X_train)


    stc = StandardScaler()

    X_train_scaled = stc.fit_transform(X_train)
    X_test_scaled = stc.transform(X_test)
    xgbooster = XGBClassifier()
    svc = SVC()
    gradient_boost = GradientBoostingClassifier()
    forest= RandomForestClassifier()
    svc.fit(X_train_scaled,y_train)
    gradient_boost.fit(X_train_scaled,y_train)
    forest.fit(X_train_scaled,y_train)
    xgbooster.fit(X_train_scaled,y_train)


    y_pred = svc.predict(X_test_scaled)
    y_pred_grad = gradient_boost.predict(X_test_scaled)
    y_pred_forest = forest.predict(X_test_scaled)
    y_pred_booster = xgbooster.predict(X_test_scaled)


    st.write(accuracy_score(y_test,y_pred))
    st.write(accuracy_score(y_test,y_pred_grad))
    st.write(accuracy_score(y_test,y_pred_forest))
    st.write(accuracy_score(y_test,y_pred_booster))
    st.session_state.X = X
    st.session_state.y = y

# --------------------- *******Make classification models********* ---------------------------- #

# Close connection with DB
cursor.close()
conn.close()