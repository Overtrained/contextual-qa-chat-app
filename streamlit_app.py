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


# for extracting source dictionary from retriever results
def extract_sources(docs):
    sources = []
    for doc in docs:
        source = {
            "subreddit": doc.metadata["subreddit"],
            "title": doc.metadata["title"],
            "content": doc.page_content,
        }
        sources.append(source)
    return sources


# for generating formatted html to display sources
def get_sources_html(sources):
    n_sources = len(sources)
    html_out = ""
    for i, source in enumerate(sources):
        html_out += f"<p><blockquote>{source['content']}</blockquote></p>"
        html_out += f"Subreddit: {source['subreddit']}, Title: {source['title']}"
        if i < n_sources - 1:
            html_out += "<p><hr/></p>"
    return html_out


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

    # metadata filters
    with st.expander("Metadata Filter"):
        subreddits_selected = st.multiselect(
            "subreddit(s) included", options=all_subreddits, default=all_subreddits
        )

    st.markdown(
        ":computer: GitHub repo [here](https://github.com/Overtrained/contextual-qa-chat-app)"
    )

if st.session_state["openai_key_check"]:
    # # initialize retriever
    # retriever = init_retriever()
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

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message.keys():
            with st.expander("Sources"):
                sources_html = get_sources_html(message["sources"])
                st.write(sources_html, unsafe_allow_html=True)


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]
    st.session_state.memory = ConversationSummaryBufferMemory(
        llm=OpenAI(),
        memory_key="chat_history",
        input_key="human_input",
        max_token_limit=100,
        human_prefix="",
        ai_prefix="",
    )


st.sidebar.button("Clear Chat History", on_click=clear_chat_history)


# Function for generating openai response
def generate_openai_response(human_input):
    template = """You are a chatbot having a conversation with a human. 
    You are an expert on the finance opinion from the collective reddit community.
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use five sentences maximum and explain your reasoning. 

    {context}

    {chat_history}

    Question: {human_input}

    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "human_input", "chat_history"], template=template
    )

    # stuff chain
    llm = OpenAI(model_name=selected_model, temperature=temperature, top_p=top_p)
    qa_chain = load_qa_chain(
        llm=llm,
        chain_type="stuff",
        prompt=QA_CHAIN_PROMPT,
        verbose=True,
        memory=st.session_state["memory"],
    )

    # generate response
    similar_docs = vectorstore.max_marginal_relevance_search(
        human_input, k=4, fetch_k=30, lambda_mult=0.5
    )
    result = qa_chain({"input_documents": similar_docs, "human_input": human_input})

    # return result
    return result


# User-provided prompt
if prompt := st.chat_input(disabled=not openai_key):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = generate_openai_response(prompt)
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
