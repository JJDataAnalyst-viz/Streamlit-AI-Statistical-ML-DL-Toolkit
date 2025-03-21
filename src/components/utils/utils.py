import streamlit as st
import random
from streamlit_avatar import avatar
from streamlit_extras.let_it_rain import rain

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


def rain_func():
        ai_emojis = ["🤖", "🧠", "📊"]
        rain(
            emoji=random.choice(ai_emojis),
            font_size=54,
            falling_speed=5,
            animation_length="infinite",
        )
  
