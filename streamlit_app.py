import streamlit as st
import os
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationSummaryBufferMemory
import pinecone
from utils import *


# vectorstore connection
@st.cache_resource
def init_vectorstore():
    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

    # connect to pinecone vectorstore
    pinecone.init(environment=st.secrets["PINECONE_ENV"])
    embeddings = OpenAIEmbeddings()
    vectorstore = Pinecone.from_existing_index(
        index_name="reddit-finance", embedding=embeddings
    )
    return vectorstore


# App title
st.set_page_config(page_title="Reddit Finance Chatbot 🤑💬")

# initialize OpenAI Credential Check
if "openai_key_check" not in st.session_state.keys():
    st.session_state["openai_key_check"] = False

# OpenAI Credentials
with st.sidebar:
    st.title("Reddit Finance Chatbot 🤑💬")
    if "OPENAI_API_KEY" in st.secrets:
        st.success("API key already provided!", icon="✅")
        openai_key = st.secrets["OPENAI_API_KEY"]
        st.session_state["openai_key_check"] = True
    else:
        openai_key = st.text_input("Enter OpenAI API key:", type="password")
        if not (openai_key.startswith("sk-") and len(openai_key) == 51):
            st.warning("Please enter your credentials!", icon="⚠️")
        else:
            st.success("Proceed to entering your prompt message!", icon="👉")
            st.session_state["openai_key_check"] = True

    # model and model parameter selections
    with st.expander("Models and Parameters"):
        selected_model = st.selectbox(
            "Choose OpenAI model",
            ["gpt-3.5-turbo", "gpt-4"],
            index=0,
            key="selected_model",
        )
        temperature = st.slider(
            "temperature", min_value=0.01, max_value=5.0, value=0.7, step=0.01
        )
        top_p = st.slider("top_p", min_value=0.01, max_value=1.0, value=0.9, step=0.01)
        model_kwargs = {
            "model_name": selected_model,
            "temperature": temperature,
            "top_p": top_p,
        }

    # metadata filters
    with st.expander("Metadata Filter"):
        subreddits_selected = st.multiselect(
            "subreddit(s) included", options=all_subreddits, default=all_subreddits
        )
        metadata_filter = {"subreddit": {"$in": subreddits_selected}}

    # reset history in session state and memory object
    st.button("Clear Chat History", on_click=clear_chat_history)

    # repo link
    st.markdown(
        ":computer: GitHub repo [here](https://github.com/Overtrained/contextual-qa-chat-app)"
    )

if st.session_state["openai_key_check"]:
    # initialize vectorstore
    vectorstore = init_vectorstore()
    # initialize memory
    if "memory" not in st.session_state.keys():
        st.session_state.memory = ConversationSummaryBufferMemory(
            llm=OpenAI(),
            memory_key="chat_history",
            input_key="human_input",
            max_token_limit=100,
            human_prefix="",
            ai_prefix="",
        )

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message.keys():
            with st.expander("Sources"):
                sources_html = get_sources_html(message["sources"])
                st.write(sources_html, unsafe_allow_html=True)


# User-provided prompt
if prompt := st.chat_input(disabled=not openai_key):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = generate_openai_response(
                prompt, vectorstore, model_kwargs, metadata_filter
            )
            st.write(result["output_text"])
            sources = extract_sources(result["input_documents"])
            with st.expander("Sources"):
                sources_html = get_sources_html(sources)
                st.write(sources_html, unsafe_allow_html=True)

    message = {
        "role": "assistant",
        "content": result["output_text"],
        "sources": sources,
    }
    st.session_state.messages.append(message)
