import streamlit as st
from part1_agent import ask_agent
from rag_agent import ask_rag

st.set_page_config(page_title="TP LangChain", layout="wide")

st.title("TP LangChain : Agent simple + RAG")

mode = st.sidebar.selectbox(
    "Choisir une partie",
    ["Partie 1 : Agent simple", "Partie 2 : RAG"]
)

if mode == "Partie 1 : Agent simple":
    st.subheader("Agent simple avec mémoire, tools et middleware")

    if "messages_agent" not in st.session_state:
        st.session_state.messages_agent = []

    for msg in st.session_state.messages_agent:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Pose une question à l'agent...")

    if user_input:
        st.session_state.messages_agent.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        response = ask_agent(user_input)

        st.session_state.messages_agent.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

else:
    st.subheader("Chatbot RAG Agentique")

    if "messages_rag" not in st.session_state:
        st.session_state.messages_rag = []

    for msg in st.session_state.messages_rag:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    rag_input = st.chat_input("Pose une question sur les documents...")

    if rag_input:
        st.session_state.messages_rag.append({"role": "user", "content": rag_input})
        with st.chat_message("user"):
            st.markdown(rag_input)

        response = ask_rag(rag_input)

        st.session_state.messages_rag.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)