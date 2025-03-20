from sklearn.datasets import make_classification
import streamlit as st


data = make_classification()

st.write(data[0],data[1])