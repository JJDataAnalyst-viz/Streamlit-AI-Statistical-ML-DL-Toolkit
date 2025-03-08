import streamlit as st

def home():
    st.title("Statistical Analysis & Machine Learning App")

    # Welcome message
    st.markdown("""
        Welcome to the **Statistical Analysis & ML/DL Toolkit**.
        This app provides tools for performing various **statistical tests**, analyzing **distributions**, and implementing **machine learning** and **deep learning models**.
        Whether you are working with hypothesis testing or exploring predictive models, this app has you covered.
    """)

    # Key Features Section
    st.header("🔬 Key Features")

    st.markdown("""
    - **Statistical Tests**  
        Perform various statistical tests including:
        - **Shapiro-Wilk Test**: Test for normality in your data.
        - **T-test**: Compare the means of two samples.
        - **ANOVA**: Test for differences between more than two groups.

    - **Distributions**  
        Explore and analyze different types of distributions:
        - **Normal Distribution**
        - **Binomial Distribution**
        - **Poisson Distribution**
        - **Exponential Distribution**
    
    - **Machine Learning Models**  
        Implement common ML algorithms:
        - **Linear Regression**
        - **Logistic Regression**
        - **Decision Trees**
        - **Random Forests**
        - **K-Means Clustering**
    
    - **Deep Learning Models**  
        Explore powerful DL techniques:
        - **Neural Networks (NN)**
        - **Convolutional Neural Networks (CNN)**
        - **Recurrent Neural Networks (RNN)**
    """)

    # How it Works Section
    st.header("📈 How It Works")
    st.markdown("""
    1. **Select Your Test/Model**  
    Choose from a variety of statistical tests or ML/DL models.
    
    2. **Input Your Data**  
    Upload your dataset or enter the data manually.
    
    3. **Run the Analysis**  
    The app will process the data and provide you with results, including p-values, test statistics, or model predictions.
    
    4. **Visualize Results**  
    View interactive visualizations for distributions, test results, and model performance.
    """)

    # Key Tests and Models Section
    st.header("🔎 Key Statistical Tests & Models")

    # Shapiro-Wilk Test
    st.subheader("1. Shapiro-Wilk Test")
    st.markdown("""
        Used for testing the normality of your dataset. It returns a p-value to determine if the data follows a normal distribution.
    """)

    # T-test
    st.subheader("2. T-test")
    st.markdown("""
        Compares the means of two independent groups to determine if there is a significant difference between them.
    """)

    # ANOVA
    st.subheader("3. ANOVA (Analysis of Variance)")
    st.markdown("""
        Used to test for significant differences between three or more groups or variables.
    """)

    # Distributions
    st.subheader("4. Distributions")
    st.markdown("""
        Explore and analyze the characteristics of various statistical distributions like:
        - Normal, Binomial, Poisson, etc.
    """)

    # Machine Learning Models
    st.subheader("5. Machine Learning Models")
    st.markdown("""
        Run and evaluate machine learning models to make predictions or classify data:
        - **Linear Regression** for predicting continuous values.
        - **Logistic Regression** for binary classification.
        - **Random Forest** and **Decision Trees** for classification tasks.
    """)

    # Deep Learning Models
    st.subheader("6. Deep Learning Models")
    st.markdown("""
        Explore deep learning models for advanced tasks:
        - **Neural Networks (NN)**: Used for both classification and regression tasks.
        - **Convolutional Neural Networks (CNN)**: Best for image and video data.
        - **Recurrent Neural Networks (RNN)**: Perfect for sequential data (e.g., time series).
    """)

    # Visualizations Section
    st.header("📊 Visualizations")

    st.markdown("""
    - **Interactive Charts**  
    Display interactive visualizations of your statistical tests and model predictions.
    
    - **Distributions**  
    View the fitted distributions to compare how well the data matches different statistical models.
    
    - **Model Performance**  
    Analyze model accuracy, loss curves, confusion matrices, and more.
    """)

    # Tools & Technologies Section
    st.header("🛠️ Tools & Technologies")

    st.markdown("""
    - **Backend**: Python, SciPy, Scikit-learn, TensorFlow, Keras
    - **Frontend**: Streamlit, Plotly, Matplotlib
    - **Deployment**: Docker, Heroku (or your preferred cloud platform)
    """)

    # Help Section
    st.header("❓ Need Help?")
    st.markdown("""
    If you encounter any issues or need help with the app, feel free to reach out via our support Slack channel or check out the **FAQ section**.
    """)

    # GitHub Section
    st.header("⭐ Star the Project on GitHub")
    st.markdown("""
    If you find this app useful, give it a star on GitHub to support the development!
    """)
    st.subheader('This description was generated by AI')
    st.markdown("""
    **Application is based for learning ML,DL and Statistics so support from generating simple text is for time saving purpose **
""")

if __name__ == "__main__":
    home()
