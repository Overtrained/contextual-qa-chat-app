import streamlit as st
import os
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
import pinecone

from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationSummaryBufferMemory


@st.cache_resource
def init_retriever(openai_key):
    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

    # connect to pinecone vectorstore
    pinecone.init(environment=st.secrets["PINECONE_ENV"])
    embeddings = OpenAIEmbeddings()
    vectorstore = Pinecone.from_existing_index(
        index_name="reddit-finance", embedding=embeddings
    )

    # setup retreiver
    metadata_field_info = [
        AttributeInfo(
            name="subreddit",
            description="The subreddit or community where the content was posted.",
            type="string",
        ),
    ]
    document_content_description = "Text content and metadata from many finance communities, or subreddits, posted to reddit."
    retriever = SelfQueryRetriever.from_llm(
        OpenAI(temperature=0),
        vectorstore,
        document_content_description,
        metadata_field_info,
        verbose=True,
    )
    return retriever


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

    st.markdown(
        ":computer: GitHub repo [here](https://github.com/Overtrained/contextual-qa-chat-app)"
    )

if st.session_state["openai_key_check"]:
    # initialize retriever
    retriever = init_retriever(openai_key)
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
    Use three sentences maximum and keep the answer as concise as possible. 

    {context}

    {chat_history}

    Question: {human_input}

    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "human_input", "chat_history"], template=template
    )

    # stuff chain
    qa_chain = load_qa_chain(
        llm=OpenAI(model_name=selected_model, temperature=temperature, top_p=top_p),
        chain_type="stuff",
        prompt=QA_CHAIN_PROMPT,
        verbose=True,
        memory=st.session_state["memory"],
    )

    # generate response
    similar_docs = retriever.get_relevant_documents(human_input)
    result = qa_chain({"input_documents": similar_docs, "human_input": human_input})

    return result["output_text"]


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
