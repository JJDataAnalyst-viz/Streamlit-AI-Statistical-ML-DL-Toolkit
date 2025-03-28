# ------------------------------
# Standard Library
# ------------------------------
import yaml
import random
from components.utils import avatar_func,rain_func,links

# ------------------------------
# Modules for Connection to DB
# ------------------------------
import mysql.connector
from dotenv import load_dotenv
import os

# ------------------------------
# Libraries for Vizualization 
# ------------------------------

import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------
# Streamlit components
# ------------------------------

import streamlit as st

# ------------------------------
# DataFrame
# ------------------------------

import pandas as pd

# ------------------------------
# Scikit-learn components
# ------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier

# ------------------------------
# XGboost Library
# ------------------------------
from xgboost import XGBClassifier


class Classification_page:
    def __init__(self):
        with open('params.yaml') as f:
            params = yaml.safe_load(f)
        st.set_page_config(page_title='Classification problem',page_icon='🎯',**params['page_config'])
        self.tabs1,self.tabs2,self.tabs3,tabs4,tabs5,tabs6 =st.tabs(['Description','Data','Prediction','Statistical analysis','Plotting data','Plotting Prediction'])
        
    def main(self):
        self.tabber1()
        self.connection_db()
        self.tabber2()
        self.tabber3()
                
    
    def connection_db(self):
        load_dotenv()
    
        self.conn = mysql.connector.connect(
        host = os.getenv("HOST_car"),
        username = os.getenv("USERNAME_car"),
        password = os.getenv("PASSWORD_car"),
        port = os.getenv("PORT_car"),
        database = os.getenv("DATABASE_car")
        )
        self.cursor =  self.conn.cursor()

    

    def tabber2(self):
            df = self.df_from_db()
            with self.tabs2:
                st.dataframe(df)

                dummy = pd.get_dummies(df.loc[:,['Gender','Helmet_Used','Seatbelt_Used']],dtype=int)
                dummy = dummy.loc[:,['Gender_Female','Helmet_Used_No','Seatbelt_Used_No']]
                dff = pd.concat([df.iloc[:,[0,2,5]],dummy],axis=1)

               

                X = dff.drop(columns='Survived')
                y = dff['Survived']

                y_streamlit = pd.DataFrame(data=df['Survived']).set_index(df['Gender'])
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.y_streamlit = y_streamlit
                self.cursor.close()
                self.conn.close()

    def df_from_db(self):
        
        self.cursor.execute("SELECT * FROM car_acc")

        res = self.cursor.fetchall()

        columns = ['Age', 'Gender', 'Speed_of_Impact', 'Helmet_Used', 'Seatbelt_Used','Survived']

        df = pd.DataFrame(res, columns=columns)
        return df
    

    
    def tabber1(self):
        with self.tabs1:
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
            links()



    def tabber3(self):
        with self.tabs3:
            with st.expander('Choosen column for classification'):
                col1,col2 = st.columns(2)

                def neger(val):
                    return f"color:red;" if val < 3 else "color:green;"
                with col1:
                    st.session_state.X.style.applymap(neger)
                    st.write(st.session_state.X.style.applymap(neger))
                with col2:
                    st.write(st.session_state.y_streamlit)

            


            X_train,X_test,y_train,y_test = train_test_split(st.session_state.X, st.session_state.y,test_size=0.2)

            selection = st.segmented_control('Widget',['Train test split','Statistical Analysis'])
            if selection == 'Train test split':
                st.divider()
                st.markdown(f"""
                                <table>
                                    <tr>
                                        <td><p style='display:inline;color:#ffffff;'>X train shape</p></td>
                                        <td> <p style='display:inline;color:#ffffff;'>X test shape : </p></td>
                                        <td><p style='display:inline;color:#ffffff;'>Y train shape : </p></td>
                                        <td><p style='display:inline;color:#ffffff;'>Y test shape : </p></td>
                                    </tr>
                                    <tr>
                                        <td><p style='display:inline;color:#35a0b9;'> {X_train.shape}</p></td>
                                        <td> <p style='display:inline;color:#35a0b9;'> {X_test.shape}</p></td>
                                        <td> <p style='display:inline;color:#35a0b9;'> {y_train.shape}</p></td>
                                        <td><p style='display:inline;color:#35a0b9;'> {y_test.shape}</p></td>
                                    </tr>
                                    </table>
                            """,unsafe_allow_html=True)
            if selection == 'Statistical Analysis':
                st.write(st.session_state.X.corr())
                
                corr = st.session_state.X.corr()

                def neg(val):
                    color = 'red' if val < 0 else 'green'  
                    return f'background-color: {color}'  
                

                styled_corr_matrix = corr.style.applymap(neg)

                st.dataframe(styled_corr_matrix)
                st.markdown(f"""<table>
                                    <tr>
                                        <td></td>
                                    </tr>
                                    <tr>
                                        <td></td>
                                    </tr>
                                </table>
                                """,unsafe_allow_html=True)
        
        
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
            
           


if __name__ == "__main__":
    obj  = Classification_page() # Instantiate object and run main function
    obj.main()



    


