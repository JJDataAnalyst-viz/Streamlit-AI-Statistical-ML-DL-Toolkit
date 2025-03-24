import streamlit as st
import random
from streamlit_avatar import avatar
from streamlit_extras.let_it_rain import rain
from streamlit_extras.mention import mention
import os


def avatar_func():
    with st.sidebar:
        avatar(
            [
                {
                    "url": "https://dagshub.com/avatars/64865?s=290'",
                    "size": 40,
                    "title": "SciOpsEngineer",
                    "caption": "Welcome on my site!",
                    "key": "avatar1",
                }
            ]
        )
def links():
    with st.sidebar:
        mention(
            label="Check my Dagshub!",
            icon="https://yt3.googleusercontent.com/Ql4cd1QPeCvwNgUlZ2zdn9W70SDeNS4FlAp_2AgaNSUuXJTeJLS9UVRwDJwBhGSHP8cqW63Q=s900-c-k-c0x00ffffff-no-rj", 
            url = 'https://dagshub.com/Jakub_Jedrych'
        )
        mention(
            label="Check my Linkedin!",
            icon="https://static.vecteezy.com/system/resources/previews/018/930/480/non_2x/linkedin-logo-linkedin-icon-transparent-free-png.png",  
            url="https://www.linkedin.com/in/jakub-jedrych/",
        )


def rain_func():
        ai_emojis = ["🤖", "🧠", "📊"]
        rain(
            emoji=random.choice(ai_emojis),
            font_size=54,
            falling_speed=5,
            animation_length="infinite",
        )
  
