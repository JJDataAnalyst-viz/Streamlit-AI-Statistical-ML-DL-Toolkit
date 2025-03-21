from sklearn.datasets import make_classification
from abc import abstractmethod,ABC
import seaborn as sns
import os
import pandas as pd

class GenerateData(ABC):
    def __init__(self):
        pass

    @abstractmethod    
    def plot_data(self):
        pass

    @abstractmethod
    def make_data(self):
        pass

class ClassificationData(GenerateData):
    
    def __init__(self):
        pass
    # ,n_features,n_informative,n_redundant,n_repeated,*,RandomState=42
    def plot_data(self):
        return super().plot_data()
    def make_data(self,n_samples,):
        X,y = make_classification(n_samples)
        return X,y
class RegressionData(GenerateData):

    def __init__(self):
        pass

class ClusterData(GenerateData):

    def __init__(self):
        pass
class Iris(GenerateData):

    def __init__(self,path):
        self.path = path


    def plot_data(self):
        return super().plot_data()

    def make_data(self):
        df = pd.read_csv(self.path,header=None,encoding='utf-8')
        return df