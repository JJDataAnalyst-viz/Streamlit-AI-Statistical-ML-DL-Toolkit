import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import pandas as pd

import statsmodels.api as sm

class LinearRegression:
    def __init__(self):
        pass
    def main(self):
        st.subheader('Linear Regression')
        LinearRegression.custom_dataset(self)

    def custom_dataset(self):
        n_samples = st.number_input('Number of samples',min_value=10,max_value=10000000)
        n_features = st.number_input('Number of features',min_value=1,max_value=4)
        features = dict()
        match n_features:
            case 1:
                x,y = make_regression(n_samples=n_samples,n_features=n_features)
                
            case 2:
                X,y = make_regression(n_samples=n_samples,n_features=n_features)
            case _:
                X,y = make_regression(n_samples=n_samples,n_features=n_features)
                st.write(pd.DataFrame(data= X,columns=[f'x{i}' for i in range(n_features)]))
                    
                st.write(features)


if __name__=="__main__":
    obj = LinearRegression()
    obj.main()