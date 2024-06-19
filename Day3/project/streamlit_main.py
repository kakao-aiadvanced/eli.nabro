import os

import streamlit as st

if "wide" not in st.session_state:
    st.set_page_config(layout="wide", initial_sidebar_state='collapsed')
    st.session_state["wide"] = True

st.markdown("""
<style>
.st-emotion-cache-12fmjuu {
    padding-top: 0.5rem;
}
</style>

""", unsafe_allow_html=True)

st.title(':hammer_and_pick: KAKAO - Ai Advanced 프로젝트 :gear: :shopping_trolley: :firecracker: :100:')

question_col, answer_col = st.columns([0.50, 0.50])
openai_api_key = question_col.text_input("Open Ai API key", type="password", value = os.environ.get("OPENAI_API_KEY"))
langchain_api_key = question_col.text_input("Langchain API key", type="password", value = os.environ.get("LANGCHAIN_API_KEY"))
tavily_api_key = question_col.text_input("TAVILY_API_KEY API key", type="password", value = os.environ.get("TAVILY_API_KEY"))

question = question_col.text_input("Questions")




