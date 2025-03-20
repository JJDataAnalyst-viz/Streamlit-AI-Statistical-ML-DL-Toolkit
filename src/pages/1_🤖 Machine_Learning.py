import streamlit as st
from PIL import Image
import io
import base64

import streamlit as st


st.set_page_config(page_title='Machine Learning',layout="wide",page_icon='🤖')


file = open("assets/ML.png", "rb")
contents = file.read()
img_str = base64.b64encode(contents).decode("utf-8")
buffer = io.BytesIO()
file.close()
img_data = base64.b64decode(img_str)
img = Image.open(io.BytesIO(img_data))
resized_img = img.resize((140, 115))  # x, y
resized_img.save(buffer, format="PNG")
img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

st.markdown(
        f"""
        <style>
            [data-testid="stSidebarNav"] {{
                background-image: url('data:image/png;base64,{img_b64}');
                background-repeat: no-repeat;
                padding-top: 50px;
                background-position: 200px -15px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
page_element = """
<style>
[data-testid="stAppViewContainer"] {
  background-image: url("https://images.steamusercontent.com/ugc/879748616164108107/8F44EE6DAFB4F4E2469AA4947059A09E1A78E93C/?imw=5000&imh=5000&ima=fit&impolicy=Letterbox&imcolor=%23000000&letterbox=false");
  background-size: cover;
  
}

[data-testid="stAppViewContainer"]:before {
  content: "";
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.6); /* Adjust the opacity (0.5) to make it darker */
  z-index: 1;
}

[data-testid="stHeader"] {
  background-color: rgba(0, 0, 0, 0);
}

[data-testid="stMarkdownContainer"] {
  position: relative;
  z-index: 2;
}
</style>
"""

st.markdown(page_element, unsafe_allow_html=True)


st.header('Machine Learning')



