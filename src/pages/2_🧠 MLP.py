import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import streamlit as st

st.set_page_config(page_title=" Multi Layer Perceptron", page_icon="🧠", layout="wide")

class Perceptron:
    '''
    Classificator -- Perceptron

    Parameters 
    ---------
    eta : float
        coefficient with range between 0.0 and 1.0
    n_iter : int
        parameter defines how many times you are updating your weights

    Atributes 
    ---------
    w_ : array
        Updated weights after training.
    errors : list
        Number of classification errors in each epoch.
   '''
    def __init__(self,eta=0.01,random_state=42,n_iter=50):
        self.eta = eta
        self.random_state=random_state
        self.n_iter = n_iter

    def fit(self,X,y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0,scale=0.1,size=1+X.shape[1])

        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi,target in zip(X,y):
                update = self.eta*(target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    def net_input(self,X):
        return np.dot(X,self.w_[1:]) + self.w_[0]
    
    def predict(self,X):
        return np.where(self.net_input(X) >= 0,1,-1)





class MLP_Pred:

        tabs1,tabs2 = st.tabs(['Data','Prediction'])
        with tabs1:
            col1,col2 = st.columns(2)

            with col1:
                data = make_classification(n_samples=1000,n_features=4)
                X = data[0]
                y = np.where(data[1] == 0,-1,1)
                
                st.write(X)
            with col2:
                st.write(y)

            perceptron = Perceptron()
            perceptron.fit(X,y)
            predicted = perceptron.predict(X)
            st.write(accuracy_score(y,predicted))
        with tabs2:
            st.write('Hello')

if __name__ == "__main__":
     obj = MLP_Pred()