from PIL import Image
import io
import base64
from components.utils import avatar_func,links,create_background_image,create_image
import streamlit as st

class MachineLearning:
    
    def __init__(self):
        pass
        st.set_page_config(page_title='Machine Learning',layout="wide",page_icon='🤖')
        st.header('Machine Learning')

    def main(self):
        self.sidebar_components()
        self.set_images()
        self.markdown_descripiton()
    def sidebar_components(self):
          avatar_func()
          links()
    def set_images(self):
        create_image("assets/ML.png",140,115,width=200,height=-15)
        st.markdown(create_background_image("https://images.steamusercontent.com/ugc/879748616164108107/8F44EE6DAFB4F4E2469AA4947059A09E1A78E93C/?imw=5000&imh=5000&ima=fit&impolicy=Letterbox&imcolor=%23000000&letterbox=false"),unsafe_allow_html=True)
    def markdown_descripiton(self):
        st.markdown("""
            **Machine Learning with**  <img src= 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png' width="40" height="40" style="vertical-align:middle" alt="Python logo"> </br>
          

            Welcome to my app which consist of Machine/Deep learning, AI and Statistics! Here, I explore and create powerful models that use cutting-edge technologies to solve complex problems. Whether it's working with **Supervised Learning**, **Unsupervised Learning**, **Deep Learning**, **AI** or **Statistic modeling**. 
            </br>I utilize libraries like:

            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/NumPy_logo_2020.svg/1200px-NumPy_logo_2020.svg.png" width="110" height="50" style="vertical-align:middle" alt="NumPy logo"> for mathematical computing  
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Pandas_logo.svg/2560px-Pandas_logo.svg.png" width="110" height="50" style="vertical-align:middle" alt="Pandas logo"> for data manipulation  
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/Created_with_Matplotlib-logo.svg/2048px-Created_with_Matplotlib-logo.svg.png" width="40" height="40" style="vertical-align:middle" alt="Matplotlib logo"> for data visualization  
            <img src="https://cdn.worldvectorlogo.com/logos/seaborn-1.svg" width="40" height="40" style="vertical-align:middle" alt="Seaborn logo"> for data visualization  
            <img src="https://companieslogo.com/img/orig/ALTR-c0246b7f.png?t=1720244490" width="30" height="30" style="vertical-align:middle" alt="Altair logo"> for data visualization  
            <img src="https://numfocus.org/wp-content/uploads/2017/11/bokeh-logo-300.png" width="50" height="50" style="vertical-align:middle" alt="Bokeh logo"> for data visualization  
            <img src="https://ml.globenewswire.com/Resource/Download/54ca9baa-43ae-4b0d-bcc3-5dcde3ab7ce0" width="50" height="40" style="vertical-align:middle" alt="Plotly logo"> for interactive visualizations
                    
            <img src="https://numfocus.org/wp-content/uploads/2017/11/scipy_logo300x300.png" width="40" height="40" style="vertical-align:middle" alt="Scipy logo"> for advanced statistics
            
            <img src="https://raw.githubusercontent.com/pymc-devs/brand/main/pymc/pymc_logos/PyMC_banner.svg" width="40" height="40" style="vertical-align:middle" alt="PyMC logo"> for advanced statistics
            
            <img src="https://images.seeklogo.com/logo-png/10/2/penguin-logo-png_seeklogo-107304.png" width="40" height="40" style="vertical-align:middle" alt="Pingouin logo"> for advanced statistics
                    
            <img src="https://docs.sympy.org/latest/_images/sympy.svg" width="40" height="40" style="vertical-align:middle" alt="Sympy logo"> for creating advanced formulas   
            <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" width="50" height="50" style="vertical-align:middle" alt="Scikit-learn logo"> for machine learning models  
            <img src="https://media2.dev.to/dynamic/image/width=1000,height=420,fit=cover,gravity=auto,format=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2F7w8rh2oj5arc1epo2sls.png" width="100" height="40" style="vertical-align:middle" alt="XGBoost logo"> for gradient boosting techniques  
            <img src="https://lightgbm.readthedocs.io/en/stable/_static/LightGBM_logo_grey_text.svg" width="100" height="40" style="vertical-align:middle" alt="LightGBM logo"> for gradient boosting techniques  
            <img src="https://datasolut.com/wp-content/uploads/2019/09/keras-logo-2018-large-1200.png" width="90" height="30" style="vertical-align:middle" alt="Keras logo">  for deep learning applications 
            
            <img src="https://img.icons8.com/arcade/512/pytorch.png" width="50" height="50" style="vertical-align:middle" alt="PyTorch logo">  for deep learning applications 
                    
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/1200px-Tensorflow_logo.svg.png" width="40" height="40" style="vertical-align:middle" alt="Tensorflow logo"> for deep learning applications 
            
                    
                    
            I create models like **Decision Trees**, **Random Forests**, **SVMs**, **KNN**, and more, aiming to provide deep insights through predictive analytics.

            ### My expertise includes:
            - **Machine Learning Algorithms**: Supervised & Unsupervised Learning.
            - **Data Processing**: Data wrangling, cleaning, and feature engineering using **Pandas** and **NumPy**.
            - **Visualization**: Insightful visualizations using **Matplotlib**, **Seaborn**, **Plotly**.
            - **Advanced Techniques**: **Deep Learning** with **PyTorch**, **XGBoost** for advanced models.
            - **Statistical Analysis**: Hypothesis testing, regression models, and more.

            By combining these tools, I tackle a variety of data challenges, from predicting outcomes to uncovering hidden patterns in large datasets.

            Let's unlock the power of **Machine Learning** together! 🚀

            ### Tools I frequently use:
            - **NumPy** and **Pandas** for data manipulation
            - **Scikit-learn** and **XGBoost** for building powerful machine learning models
            - **Matplotlib**, **Seaborn**, and **Plotly** for visualization
            - **Scipy** and **Statsmodels** for statistical analysis
            - **PyTorch** for deep learning and neural networks 🧠
            - **Streamlit** for creating interactive data-driven apps 🖥️

            Let's explore the fascinating world of data science and machine learning! 🔍💡
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    obj = MachineLearning()
    obj.main()

