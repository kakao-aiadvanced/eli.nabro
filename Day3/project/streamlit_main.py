import os

import streamlit as st

from chain_executor import execute

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
doc_urls_text: str = question_col.text_area("Initial docs",value="""
https://lilianweng.github.io/posts/2023-06-23-agent/
https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/
https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/
""".strip())
doc_urls = [line.strip() for line in doc_urls_text.split('\n')]

user_question = question_col.text_input("User Question")

if user_question:
    result = execute(user_question, doc_urls, openai_api_key, langchain_api_key, tavily_api_key)
    print(result)
    answer_col.write(result)


