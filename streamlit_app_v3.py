import streamlit as st
import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# App title
st.set_page_config(page_title="Reddit Finance Chatbot 🤑💬")

# OpenAI Credentials
with st.sidebar:
    st.title("Reddit Finance Chatbot 🤑💬")
    if "OPENAI_API_KEY" in st.secrets:
        st.success("API key already provided!", icon="✅")
        openai_key = st.secrets["OPENAI_API_KEY"]
    else:
        openai_key = st.text_input("Enter OpenAI API key:", type="password")
        if not (openai_key.startswith("sk-") and len(openai_key) == 51):
            st.warning("Please enter your credentials!", icon="⚠️")
        else:
            st.success("Proceed to entering your prompt message!", icon="👉")

    # Refactored from https://github.com/a16z-infra/llama2-chatbot
    st.subheader("Models and parameters")
    selected_model = st.sidebar.selectbox(
        "Choose OpenAI model", ["gpt-3.5-turbo", "gpt-4"], index=0, key="selected_model"
    )
    temperature = st.sidebar.slider(
        "temperature", min_value=0.01, max_value=5.0, value=0.7, step=0.01
    )
    top_p = st.sidebar.slider(
        "top_p", min_value=0.01, max_value=1.0, value=0.9, step=0.01
    )

    st.markdown(
        ":computer: GitHub repo [here](https://github.com/Overtrained/contextual-qa-chat-app)"
    )

os.environ["OPENAI_API_KEY"] = openai_key

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]


st.sidebar.button("Clear Chat History", on_click=clear_chat_history)


# Function for generating openai response
def generate_openai_response(prompt_input):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"

    prompt = PromptTemplate.from_template(
        string_dialogue + "/n/n {prompt_input} Assistant: "
    )
    llm = OpenAI(model_name=selected_model, temperature=temperature, top_p=top_p)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    output = llm_chain.run(prompt_input)

    return output


# User-provided prompt
if prompt := st.chat_input(disabled=not openai_key):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_openai_response(prompt)
            placeholder = st.empty()
            full_response = ""
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
