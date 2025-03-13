import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import seaborn as sns
import numpy as np
import mplcyberpunk

plt.style.use("cyberpunk")
class LinearRegression_model:
    def __init__(self,random_state: int = 42):
        self.random_state = random_state

    def main(self):
        np.random.seed(self.random_state)
        st.subheader('Linear Regression')
        LinearRegression_model.custom_dataset()
        

    @classmethod
    def custom_dataset(self):
        n_samples,n_features,noise = LinearRegression_model.input_df(self)
        col1,col2 = st.columns(2)
        match n_features:
            case 1:
                with col1:
                    df =LinearRegression_model.concat_dataframe(self,n_samples,n_features,noise)
                    st.dataframe(df,width=600, height=400,use_container_width = False)
                    if "df" not in st.session_state:
                        st.session_state.df = df
                    if st.button('Make Train Test Split & Scale Data'):
                        scaler_x = StandardScaler()
                        scaler_y = StandardScaler()
                        X = df.iloc[:,:-1].values
                        y = np.reshape(df.iloc[:,-1].values,shape=(-1,1))
                        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
                        X_train_scaled = scaler_x.fit_transform(X_train)
                        X_test_scaled = scaler_x.transform(X_test)
       
                        st.session_state.X_train_scaled = X_train_scaled
                        st.session_state.X_test_scaled = X_test_scaled
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
               
                  
                        st.session_state.X = X
               
                        st.session_state.y = y

                        subcol1,subcol2,subcol3,subcol4 = st.columns(4)
                        with subcol1:
                            st.write(X_train_scaled)
                        with subcol2:
                            st.write(X_test_scaled)
                        with subcol3:
                            st.write(y_train)
                        with subcol4:
                            st.write(y_test)
                with col2:
                    pairplot_fig = sns.pairplot(df,aspect=12.3/8.3,diag_kind='kde',height=1.5,kind='kde')
                    
                    
                    st.pyplot(pairplot_fig,use_container_width = True)
                    
                    if st.button('Predict'):
                        if  ('X_test_scaled' in st.session_state) & ('X_train_scaled' in st.session_state):
                           regressor = LinearRegression()
                           regressor.fit(st.session_state.X_train_scaled,st.session_state.y_train)
                           predicted =  regressor.predict(st.session_state.X_test_scaled)

                           fig, ax = plt.subplots()
                           fig.set_size_inches(6,6)

                        
                         

                           mplcyberpunk.add_glow_effects()
                           ax.plot(st.session_state.X_test_scaled, predicted, color='red', label='Regression Line')


              
                
                        #    ax.scatter(st.session_state.X_train_scaled ,st.session_state.y)

                           ax.scatter(st.session_state.X_test_scaled, st.session_state.y_test, color='red', label='Actual Data')

                           st.pyplot(fig=fig)
                           st.write(root_mean_squared_error(st.session_state.y_test,predicted))
                           st.write(predicted,st.session_state.y_test)
                        else:
                              pass
          
               

            case 2:
                with col1:
                    df =LinearRegression_model.concat_dataframe(self,n_samples,n_features,noise)
                    st.dataframe(df,width=600, height=400,use_container_width = False)
                    if st.button('Make Train Test Split & Scale Data'):
                        scaler_x = StandardScaler()
                        scaler_y = StandardScaler()
                        X = df.iloc[:,:-1].values
                        y = np.reshape(df.iloc[:,-1].values,shape=(-1,1))
                        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
                        X_train_scaled = scaler_x.fit_transform(X_train)
                        X_test_scaled = scaler_x.transform(X_test)
             
                        st.session_state.X_train_scaled = X_train_scaled
                        st.session_state.X_test_scaled = X_test_scaled
                        st.session_state.X_test_scaled_X1 = X_test_scaled[:,-1]
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test

                        st.session_state.X_1 = X[:,-1]
                        st.session_state.X_2 = X[:,0]
                        st.session_state.y = y

                        subcol1,subcol2,subcol3,subcol4 = st.columns(4)
                        with subcol1:
                            st.write(X_train_scaled)
                        with subcol2:
                            st.write(X_test_scaled)
                        with subcol3:
                            st.write(y_train)
                        with subcol4:
                            st.write(y_test)
                with col2:
                    pairplot_fig = sns.pairplot(df,aspect=12.3/8.3,diag_kind='kde',height=1.5,kind='kde')
                    
                    
                    st.pyplot(pairplot_fig,use_container_width = True)
                   
                    if st.button('Predict'):
                        if  ('X_test_scaled' in st.session_state) & ('X_train_scaled' in st.session_state):
                           regressor = LinearRegression()
                           regressor.fit(st.session_state.X_train_scaled,st.session_state.y_train)
                           predicted =  regressor.predict(st.session_state.X_test_scaled)

                           st.write(root_mean_squared_error(st.session_state.y_test,predicted))
                           fig, ax = plt.subplots()
                           fig.set_size_inches(6,6)


                        
                           ax.scatter(st.session_state.X_1,st.session_state.y)
                           ax.plot( st.session_state.X_test_scaled[:,0],predicted, color='red', label='Regression Line')
                           st.pyplot(fig=fig)
                         
                        
                        else:
                              pass

                    
            case _:
                df =LinearRegression_model.concat_dataframe(self,n_samples,n_features,noise)
                st.dataframe(df,width=30, height=3,use_container_width = False)
            
    def concat_dataframe(self,n_samples,n_features,noise):
        X,y = make_regression(n_samples=n_samples,n_features=n_features,noise=noise)
        x_data = pd.DataFrame(data= X,columns=[f'x{i}' for i in range(n_features)])
        y_data = pd.DataFrame(data=y,columns=['y']) 
        df = pd.concat([x_data,y_data],axis=1)
        return df

    def input_df(self):
        n_samples = st.number_input('Number of samples',min_value=10,max_value=10000000)
        n_features = st.number_input('Number of features',min_value=1,max_value=4)
        noise = st.slider('Give noise for the data',min_value=0.0,max_value=50.0)
     
        return n_samples,n_features,noise


if __name__=="__main__":
    obj = LinearRegression_model()
    obj.main()