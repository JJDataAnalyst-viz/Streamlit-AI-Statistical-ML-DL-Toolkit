from sklearn.datasets import make_classification
import streamlit as st
from components.utils import avatar_func,links

data = make_classification()
avatar_func()
links()
st.write(data[0],data[1])