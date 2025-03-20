# Libraries for connection with DB
import mysql.connector
from dotenv import load_dotenv
import os

# App
import streamlit as st

# DataFrame
import pandas as pd

# Scikit-learn


st.set_page_config(page_title='Classification problem',page_icon='🎯')



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

st.dataframe(df)



# --------------------- *******Make classification models********* ---------------------------- #


dummy = pd.get_dummies(df.loc[:,['Gender','Helmet_Used','Seatbelt_Used']],dtype=int)
dummy = dummy.loc[:,['Gender_Female','Helmet_Used_No','Seatbelt_Used_No']]
dff = pd.concat([df.iloc[:,[0,2,5]],dummy],axis=1)


st.dataframe(dff)
# --------------------- *******Make classification models********* ---------------------------- #

# Close connection with DB
cursor.close()
conn.close()