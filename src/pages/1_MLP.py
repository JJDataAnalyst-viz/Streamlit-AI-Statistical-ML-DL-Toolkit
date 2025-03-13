import numpy as np
from sklearn.datasets import make_classification
import streamlit as st
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


data = make_classification(n_samples=1000,n_features=4)

X = data[0]
y = data[1]

st.write(X)
st.write(y)