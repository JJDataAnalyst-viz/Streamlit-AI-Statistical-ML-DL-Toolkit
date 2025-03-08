import streamlit as st

from OOP_statistical_app import *
from home_app import home

st.set_page_config(layout="wide")
def main():

    menu = ['Home','Statistical analysis','About']
    choice = st.sidebar.selectbox('Menu',menu)
    if choice == 'Home':
        home()
    if choice == 'Statistical analysis':
            statistical_analysis()
if __name__ =="__main__":
    main()  

