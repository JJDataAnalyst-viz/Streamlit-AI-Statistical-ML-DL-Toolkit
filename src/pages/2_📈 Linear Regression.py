import streamlit as st
from pandas import DataFrame
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
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from sklearn.decomposition import PCA
from components.utils import avatar_func,links


plt.style.use("cyberpunk")
st.set_page_config(page_title='Linear Regression',layout="wide",page_icon="📈")
avatar_func()
links()
class LinearRegression_model:

    '''
    Class for predicting custom dataset using Linear Regression model
    
    '''
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
                    df = LinearRegression_model.make_df(self,n_samples,n_features,noise)
                    if "df" not in st.session_state:
                        st.session_state.df = df
                    if st.button('Make Train Test Split & Scale Data'):
                        X,y,X_train_scaled,X_test_scaled,y_train,y_test = LinearRegression_model.train_test_scale_data(self,df)
                
                        LinearRegression_model.make_session_state_data(self,X,y,X_train_scaled,X_test_scaled,y_train,y_test)

                        LinearRegression_model.show_splitted_data(self,X_train_scaled,X_test_scaled,y_train,y_test)


                with col2:
                    PairPlot.pairplot_chart(self,df)
                    
                    if st.button('Predict'):
                        if  ('X_test_scaled' in st.session_state) & ('X_train_scaled' in st.session_state):
                           predicted = LinearRegression_model.predict_custom_dataset(self,X_train=st.session_state.X_train_scaled,X_test=st.session_state.X_test_scaled,y_train=st.session_state.y_train)

                           fig, ax = plt.subplots()
                           fig.set_size_inches(6,6)
                   
                           ax.plot(st.session_state.X_test_scaled, predicted, color='red', label='Regression Line')

                           ax.scatter(st.session_state.X_test_scaled, st.session_state.y_test, color='red', label='Actual Data')

                           st.pyplot(fig=fig)
                           st.write(root_mean_squared_error(st.session_state.y_test,predicted))
                          
                        else:
                              pass
          
               

            case 2:
                with col1:
                    df = LinearRegression_model.make_df(self,n_samples,n_features,noise)
                    with st.expander('Make Train Test Split & Scale Data'):
                        X,y,X_train_scaled,X_test_scaled,y_train,y_test = LinearRegression_model.train_test_scale_data(self,df)
                
                        LinearRegression_model.make_session_state_data(self,X,y,X_train_scaled,X_test_scaled,y_train,y_test)

                        LinearRegression_model.show_splitted_data(self,X_train_scaled,X_test_scaled,y_train,y_test)
                        
                        
                with col2:
        
                    PairPlot.pairplot_chart(self,df)

                    with st.expander('3D - Chart projecting relationship X-Y '):
                        if  ('X_test_scaled' in st.session_state) & ('X_train_scaled' in st.session_state):
                           predicted = LinearRegression_model.predict_custom_dataset(self,X_train=st.session_state.X_train_scaled,X_test=st.session_state.X_test_scaled,y_train=st.session_state.y_train)

                           st.write('RMSE :',round(root_mean_squared_error(st.session_state.y_test,predicted),2))
                           X0_corr_stat= stats.pearsonr(df.iloc[:,-2],df.iloc[:,-1]).statistic 
                           p_val = stats.pearsonr(df.iloc[:,-2],df.iloc[:,-1]).pvalue 
                           st.write('Pearson corr X0 :'  , round(X0_corr_stat,2),' p_value : ',round(p_val,2)) ,\
                                     'Pearson corr X1 :', stats.pearsonr(df.iloc[:,0],df.iloc[:,-1])  
                           fig, ax = plt.subplots(1,2)
                           fig.set_size_inches(6,6)


                    
                           mesh_size = .02
                           margin = 0

                           x_min, x_max = st.session_state.X_1.min() - margin, st.session_state.X_1.max() + margin
                           y_min, y_max = st.session_state.X_2.min() - margin, st.session_state.X_2.max() + margin
                           xrange = np.arange(x_min, x_max, mesh_size)
                           yrange = np.arange(y_min, y_max, mesh_size)
                           xx, yy = np.meshgrid(xrange, yrange)

                        #    pred = model.predict(np.c_[xx.ravel(), yy.ravel()])
                        #    pred = pred.reshape(xx.shape)
                           fig = px.scatter_3d(df, x=df.iloc[:,-2], y=df.iloc[:,0], z=df.iloc[:,-1])
                           fig.update_traces(marker=dict(size=5))
                         

                           st.plotly_chart(fig)

                    with st.expander('charts'):
                               fig, ax = plt.subplots(1,2)
                               fig.set_size_inches(6,6)
                               ax[0].scatter(st.session_state.X_1,st.session_state.y)
                               ax[1].scatter(st.session_state.X_2,st.session_state.y)
                               st.pyplot(fig)
              

                    
            case _:
                with col1:
                    df = LinearRegression_model.make_df(self,n_samples,n_features,noise)
                    if st.button('Make Train Test Split & Scale Data'):
                            X,y,X_train_scaled,X_test_scaled,y_train,y_test = LinearRegression_model.train_test_scale_data(self,df)
                    
                            LinearRegression_model.make_session_state_data(self,X,y,X_train_scaled,X_test_scaled,y_train,y_test)

                            LinearRegression_model.show_splitted_data(self,X_train_scaled,X_test_scaled,y_train,y_test)
                         
                with col2:
        
                    PairPlot.pairplot_chart(self,df)

                    if st.button('Predict'):
                        if  ('X_test_scaled' in st.session_state) & ('X_train_scaled' in st.session_state):
                           predicted = LinearRegression_model.predict_custom_dataset(self,X_train=st.session_state.X_train_scaled,X_test=st.session_state.X_test_scaled,y_train=st.session_state.y_train)

                           st.write(root_mean_squared_error(st.session_state.y_test,predicted))
                           fig, ax = plt.subplots()
                           fig.set_size_inches(6,6)

                           ax.scatter(st.session_state.X_1,st.session_state.y)
                           ax.plot( st.session_state.X_test_scaled[:,0],predicted, color='red', label='Regression Line')
                           st.pyplot(fig=fig)
                         
                        

 # ---------------------------------****************************----------------------------------#
 # ---------------------------------    HELPER FUNCTIONS        ----------------------------------#       

    def pca_dim_2d(self,X_array):
        pca = PCA(n_components=2)
        return pca.fit_transform(X_array),pca.explained_variance_ratio_

    def make_df(self,n_samples,n_features,noise):
        df =LinearRegression_model.concat_dataframe(self,n_samples,n_features,noise)
        st.dataframe(df,width=600, height=400,use_container_width = False)
        return df

    def train_test_scale_data(self,df : DataFrame):
            scaler_x = StandardScaler()
            
            X = df.iloc[:,:-1].values
            y = np.reshape(df.iloc[:,-1].values,shape=(-1,1))
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
            X_train_scaled = scaler_x.fit_transform(X_train)
            X_test_scaled = scaler_x.transform(X_test)

            return (X,y,X_train_scaled,X_test_scaled,y_train,y_test)
        
    def make_session_state_data(self,X,y,X_train_scaled,X_test_scaled,y_train,y_test):
            st.session_state.X_train_scaled = X_train_scaled
            st.session_state.X_test_scaled = X_test_scaled
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
                
            if X.shape[1] == 1:      
                st.session_state.X = X     
                st.session_state.y = y
            elif X.shape[1] < 3:
                st.session_state.X = X     
                st.session_state.y = y
                st.session_state.X_test_scaled_X1 = X_test_scaled[:,-1]
                st.session_state.X_1 = X[:,-1]
                st.session_state.X_2 = X[:,0]
    
             
    
    def show_splitted_data(self,X_train_scaled,X_test_scaled,y_train,y_test):
        subcol1,subcol2,subcol3,subcol4 = st.columns(4)
        with subcol1:
            st.markdown("Trained X data")  # Add a custom title
            st.dataframe(X_train_scaled) 
        with subcol2:
            st.markdown("Tested X data")
            st.write(X_test_scaled)
        with subcol3:
            st.markdown("Train labels")
            st.write(y_train)
        with subcol4:
            st.markdown("Test labels")
            st.write(y_test)


    def predict_custom_dataset(self,*,X_train,X_test,y_train):
         regressor = LinearRegression()
         regressor.fit(st.session_state.X_train_scaled,st.session_state.y_train)
         predicted =  regressor.predict(st.session_state.X_test_scaled)
         return predicted

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



class PairPlot(LinearRegression_model):
    '''Creating pairplot for data'''

    def pairplot_chart(self,df):
         pairplot_fig = sns.pairplot(df,aspect=12.3/8.3,diag_kind='kde',height=1.5,kind='kde')            
         return st.pyplot(pairplot_fig,use_container_width = True)
                   

if __name__=="__main__":
    obj = LinearRegression_model()
    obj.main()