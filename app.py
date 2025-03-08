import streamlit as st

from OOP_statistical_app import *

st.set_page_config(layout="wide")
def main():

    menu = ['Home','Statistical analysis','About']
    choice = st.sidebar.selectbox('Menu',menu)
    if choice == 'Home':
        st.subheader('Home')
        st.write('Hello')

    if choice == 'Statistical analysis':
            statistical_analysis()
if __name__ =="__main__":
    main()  

