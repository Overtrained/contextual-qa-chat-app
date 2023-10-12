import streamlit as st
import os
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationSummaryBufferMemory
import pinecone


# all values of subreddit availalbe in metadata
all_subreddits = [
    "ASX_Bets",
    "AskEconomics",
    "AusFinance",
    "Bitcoin",
    "Canadapennystocks",
    "CanadianInvestor",
    "CryptoCurrency",
    "CryptoCurrencyTrading",
    "CryptoMarkets",
    "CryptoMoonShots",
    "Daytrading",
    "ETFs",
    "Economics",
    "FinancialPlanning",
    "Forex",
    "IndiaInvestments",
    "Money",
    "StockMarket",
    "StocksAndTrading",
    "Superstonk",
    "Trading",
    "UKInvesting",
    "UKPersonalFinance",
    "ValueInvesting",
    "Wallstreetbetsnew",
    "algotrading",
    "crypto_currency",
    "dividends",
    "economy",
    "ethtrader",
    "eupersonalfinance",
    "fatFIRE",
    "finance",
    "financialindependence",
    "investing",
    "options",
    "pennystocks",
    "personalfinance",
    "povertyfinance",
    "realestateinvesting",
    "stocks",
    "thetagang",
    "wallstreetbets",
]


# reset chat history in session state and memory object
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


# Function for generating openai response
def generate_openai_response(
    human_input, vectorstore, model_kwargs, metadata_filter=None
):
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
    llm = OpenAI(**model_kwargs)
    qa_chain = load_qa_chain(
        llm=llm,
        chain_type="stuff",
        prompt=QA_CHAIN_PROMPT,
        verbose=True,
        memory=st.session_state["memory"],
    )

    # generate response
    similar_docs = vectorstore.max_marginal_relevance_search(
        human_input,
        k=4,
        fetch_k=30,
        lambda_mult=0.5,
        filter=metadata_filter,
    )
    result = qa_chain({"input_documents": similar_docs, "human_input": human_input})

    # return result
    return result


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
