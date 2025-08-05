import streamlit as st
from chatbot import chatbot_response

st.set_page_config(page_title="Mental Health Chatbot", page_icon="💬")

st.title("💬 MindMate – Mental Health Chatbot for Techies")
st.markdown("Talk about stress, burnout, impostor syndrome, or just say hello 👋")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("How are you feeling today?")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get bot reply
    bot_reply = chatbot_response(user_input)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
