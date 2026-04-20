"""Streamlit UI for the Course Assistant.

This file intentionally keeps only UI concerns and imports backend logic
from agent.py so notebook, script, and deployment stay consistent.
"""

import uuid

import streamlit as st

from agent import DOCUMENTS, build_agent


st.set_page_config(page_title="Course Assistant", page_icon="🎓", layout="wide")
st.title("🎓 Course Assistant")
st.caption("Answers course questions with routing, memory, tool support, and grounded retrieval.")


@st.cache_resource
def load_resources():
    return build_agent()


try:
    course_agent = load_resources()
except Exception as e:
    st.error(f"Failed to initialize agent: {e}")
    st.stop()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Session")
    st.write(f"Thread: {st.session_state.thread_id}")
    st.write(f"KB docs: {len(DOCUMENTS)}")
    if st.button("New conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Ask about the course...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = course_agent.ask(prompt, thread_id=st.session_state.thread_id)
            answer = result.get("answer", "Sorry, I could not generate an answer.")
            route = result.get("route", "n/a")
            faith = result.get("faithfulness", 0.0)
            sources = result.get("sources", [])

        st.write(answer)
        st.caption(f"Route: {route} | Faithfulness: {faith:.2f}")
        if sources:
            st.caption("Sources: " + ", ".join(sources))

    st.session_state.messages.append({"role": "assistant", "content": answer})
