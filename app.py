import streamlit as st

from auth.login import login
from rag.chatbot import get_answer

login()

st.title("🎓 ANITS Campus AI")

st.write(
    f"Logged in as {st.session_state.role}"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input(
    "Ask about ANITS..."
)

if question:

    st.session_state.messages.append(
        {
            "role":"user",
            "content":question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    answer = get_answer(
        question,
        st.session_state.role
    )

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )