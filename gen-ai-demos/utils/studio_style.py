import streamlit as st

def keyword_label(text):
    return f'<div style="background-color: #f0f0f0; border-radius: 5px; padding: 5px 10px; display: inline-block; margin-right: 10px; font-size: 12px;">{text}</div>'

def apply_studio_style():
    st.markdown(
        """
        <style>
           @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@300&display=swap');

            html, body, [class*="css"]  {
            font-family: 'Open Sans', sans-serif;
			}
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.image("./static/ai21logo.png",width=300, use_column_width=False)

