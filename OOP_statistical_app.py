import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy import stats
import mplcyberpunk 
import yaml
from abc import ABC,abstractmethod
       
with open('params.yaml','r') as f:
    yaml_file = yaml.safe_load(f)['distribution']

def statistical_analysis():
        st.subheader('Statistical analysis',divider='rainbow')
        statistic = Statistic()
        statistic.distribution()
        if st.button('About statistical libraries in python'):
            statistic.description_analysis()

class Statistic():
    def __init__(self):
        plt.style.use('cyberpunk')
    
    def description_analysis(self):
        return (st.text('This section provides tools and tests for statistical analysis using popular Python libraries:'),  
            st.markdown("**1. Numpy** : A fundamental package for scientific computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays."),
            st.markdown('**2. Pandas**: A powerful library used for data manipulation and analysis. It offers data structures like DataFrames, which are useful for handling and analyzing structured data.'),
            st.markdown('**3. Matplotlib**: A plotting library used for creating static, animated, and interactive visualizations in Python. It can be used to generate a variety of plots and charts.'),
            st.markdown('**4. Seaborn**: A Python visualization library based on Matplotlib that provides a high-level interface for drawing attractive and informative statistical graphics.'),
            st.markdown('**5. Pingouin**: A statistical package for Python that simplifies conducting a variety of statistical tests and provides tools for hypothesis testing, correlation analysis, regression models, etc.'),
            st.markdown('**6. Scipy**: A library used for scientific and technical computing. It builds on Numpy and provides a large number of functions for optimization, integration, interpolation, eigenvalue problems, and more.'))
           

    def choose_distribution(self):
        
        distribution_data = ['Uniform','Guassian','Lognormal','Exponential','Binomial']
        choser = st.radio('Choose a distribution for data',distribution_data,horizontal=True)
        return choser
    
    @classmethod
    def distribution(self):
        choser = Statistic.choose_distribution(self)
        match choser:
            case 'Uniform':
                Uniform.create_random_data(self)
            case 'Guassian':
                Guassian.create_random_data(self)
            case 'Lognormal':
                LogNormal.create_random_data(self)
            case 'Exponential':
                  Exponential.create_random_data(self)
            case 'Binomial':
                  Binomial.create_random_data(self)

 
    def create_random_data(self):
       '''Create distribution'''
       pass
       
        


class Guassian(Statistic):
    def __init__(self):
        pass
    def create_random_data(self):
        color = yaml_file['color_gaussian']
        col1,col2= st.columns(2)

        with col1:

            np.random.seed(42)
            loc = st.number_input('Choose your mean for normal distribution',min_value=1,max_value=10000)
            size = st.number_input('choose a size for your dstribution',min_value=10,max_value=100000000000)

        with col2:
            np.random.seed(42)
            guassian = np.random.normal(loc=loc,size=size)
                    
            fig, [[ax0,ax1],[ax2,ax3]] = plt.subplots(figsize=(10,5),ncols=2,nrows=2,sharex=False) 
              
            ax0.axvline(np.mean(guassian),linestyle='--')
            sns.histplot(guassian, bins=20, kde=True, ax=ax0, edgecolor='black')
            ax0.set_title('Histogram')
                    
            box = ax1.boxplot(guassian,sym='k+',notch=True)
            for _,line_list in box.items():
                for line in line_list:
                        line.set_color(color)
            ax1.set_title('Boxplot')
                    
            ax2.violinplot(guassian)
            ax3.ecdf(guassian)
            st.pyplot(fig=fig,use_container_width=False)
           

class Uniform(Statistic):
    def __init__(self):
        pass
    def create_random_data(self):
        color = yaml_file['color_uniform']
        col1,col2= st.columns(2)
        with col1:

            np.random.seed(42)
            low_value = st.number_input('choose a low value for distribution',min_value=1,max_value=1000)
            high_value =st.number_input('choose a high value for distribution',min_value=10,max_value=100000000000)
            size = st.number_input('choose a size for your dstribution',min_value=10,max_value=100000000000)

        with col2:
            np.random.seed(42)
            uni = np.random.uniform(low = low_value,high=high_value,size=size)
                    
            fig, [[ax0,ax1],[ax2,ax3]] = plt.subplots(figsize=(10,5),ncols=2,nrows=2,sharex=False) 
            
            ax0.axvline(np.mean(uni),linestyle='--',color=color)
            sns.histplot(uni, bins=20, kde=True, ax=ax0, edgecolor='black',color=color)
            ax0.set_title('Histogram')
                    
            box = ax1.boxplot(uni,sym='k+',notch=True)
            for _,line_list in box.items():
                for line in line_list:
                        line.set_color(color)
                ax1.set_title('Boxplot')
                    
            sns.violinplot(data=uni,ax=ax2,color=color)
            sns.ecdfplot(data=uni,color=color,ax=ax3)
            st.pyplot(fig=fig,use_container_width=False)
            if low_value > high_value:
                    st.warning('Remeber setting low value below higher gives same result ')

class LogNormal(Statistic):
    def __init__(self):
        pass
    
    def create_random_data(self):
        color = yaml_file['color_lognormal']

        col1,col2= st.columns(2)

        with col1:

            np.random.seed(42)
            mean = st.number_input('Choose a mean for distribution',min_value=1,max_value=1000000)
            sigma =st.number_input('Choose a sigma for distribution',min_value=1,max_value=10)
            size = st.number_input('Choose a size for your distribution',min_value=10,max_value=100000000000)

        with col2:
            np.random.seed(42)
            
            log = np.random.lognormal(mean = mean,sigma=sigma,size=size)
                    
            fig, [[ax0,ax1],[ax2,ax3]] = plt.subplots(figsize=(10,5),ncols=2,nrows=2,sharex=False) 
            
            ax0.axvline(np.mean(log),linestyle='--',color=color)
            sns.histplot(log, bins=20, kde=True, ax=ax0, edgecolor='black',color=color)
            ax0.set_title('Histogram')
                    
            box = ax1.boxplot(log,sym='k+',notch=True)
            for _,line_list in box.items():
                for line in line_list:
                        line.set_color(color)
                ax1.set_title('Boxplot')
                    
            sns.violinplot(data=log,ax=ax2,color=color)
            sns.ecdfplot(data=log,color=color,ax=ax3)
            st.pyplot(fig=fig,use_container_width=False)
     

class Exponential(Statistic):
    def __init__(self):
        pass

    def create_random_data(self):
        color = yaml_file['color_exponential']

        col1,col2= st.columns(2)

        with col1:

            np.random.seed(42)
            scale = st.number_input('Choose a mean for distribution',min_value=1,max_value=1000)
            size = st.number_input('Choose a size for your distribution',min_value=10,max_value=100000000000)

        with col2:
            np.random.seed(42)
            
            exponential = np.random.exponential(scale=scale,size=size)
                    
            fig, [[ax0,ax1],[ax2,ax3]] = plt.subplots(figsize=(10,5),ncols=2,nrows=2,sharex=False) 
            
            ax0.axvline(np.mean(exponential),linestyle='--',color=color)
            sns.histplot(exponential, bins=20, kde=True, ax=ax0, edgecolor='black',color=color)
            ax0.set_title('Histogram')
                    
            box = ax1.boxplot(exponential,sym='k+',notch=True)
            for _,line_list in box.items():
                for line in line_list:
                        line.set_color(color)
                ax1.set_title('Boxplot')
                    
            sns.violinplot(data=exponential,ax=ax2,color=color)
            sns.ecdfplot(data=exponential,color=color,ax=ax3)
            st.pyplot(fig=fig,use_container_width=False)
     

class Binomial(Statistic):
    def __init__(self):
        pass
    def create_random_data(self):
        color = yaml_file['color_binomial']

        col1,col2= st.columns(2)

        with col1:

            np.random.seed(42)
            
            n = st.number_input('Choose n for distribution',min_value=1,max_value=1000)
            p = st.slider('Choose p for distribution',min_value=0.0,max_value=1.0)
            size = st.number_input('Choose a size for your distribution',min_value=10,max_value=100000000000)

        with col2:
            np.random.seed(42)
            
            binomial = np.random.binomial(n=n,p=p,size=size)
                    
            fig, [[ax0,ax1],[ax2,ax3]] = plt.subplots(figsize=(10,5),ncols=2,nrows=2,sharex=False) 
            
            ax0.axvline(np.mean(binomial),linestyle='--',color=color)
            sns.histplot(binomial, bins=20, kde=True, ax=ax0, edgecolor='black',color=color)
            ax0.set_title('Histogram')
                    
            box = ax1.boxplot(binomial,sym='k+',notch=True)
            for _,line_list in box.items():
                for line in line_list:
                        line.set_color(color)
                ax1.set_title('Boxplot')
                    
            sns.violinplot(data=binomial,ax=ax2,color=color)
            sns.ecdfplot(data=binomial,color=color,ax=ax3)
            st.pyplot(fig=fig,use_container_width=False)


class Statistic_analysis(ABC):
    def __init__(self):
          pass

    @abstractmethod
    def statistical_test(self):
        pass


class Onettest(Statistic_analysis):
     def __init__(self,data):
          self.data = data

     def statistical_test(self):
          mean  = st.number_input('',1,2)
          ttest_onesample = stats.ttest_1samp(a=self.data,mean=mean)



if __name__ == "__main__":
        statistical_analysis()