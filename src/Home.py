import streamlit as st
from PIL import Image
import io
import base64
from OOP_statistical_app import *
from home_app import home
from about_app import about
import yaml


with open('params.yaml') as f:
    params = yaml.safe_load(f)
    

st.set_page_config(page_title='Home',page_icon='🏡',**params['page_config'] ) # Add versioning

file = open("assets/Home.png", "rb")
contents = file.read()
img_str = base64.b64encode(contents).decode("utf-8")
buffer = io.BytesIO()
file.close()
img_data = base64.b64decode(img_str)
img = Image.open(io.BytesIO(img_data))
resized_img = img.resize((100, 90))  
resized_img.save(buffer, format="PNG")
img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

st.markdown(
        f"""
        <style>
            [data-testid="stSidebarNav"] {{
                background-image: url('data:image/png;base64,{img_b64}');
                background-repeat: no-repeat;
                padding-top: 49px;
                background-position: 220px -15px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def main():

    menu = ['Home','Statistical analysis','About']
    choice = st.sidebar.selectbox('Menu',menu)
    if choice == 'Home':
        home()
    if choice == 'Statistical analysis':
            statistical_analysis()
    if choice == 'About':
         about()
if __name__ =="__main__":
    main()  

