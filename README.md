# Reddit Finance Chatbot 🤑💬

This chatbot allows the user to converse with the opinions of redditors across many finance related subreddits.

In particular, the knowledge base for this chatbot consists of the 5,000 highest scoring posts and top comments from a comprehensive Reddit finance HuggingFace [dataset by winddude](https://huggingface.co/datasets/winddude/reddit_finance_43_250k).  The code used to create the dataset is located in this [repo](https://github.com/getorca/ProfitsBot_V0_OLLM/tree/main/ds_builder).

## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://reddit-finance-chat.streamlit.app)

## Prerequisite libraries

```
langchain
openai
streamlit
pinecone-client
lark
tiktoken
```

## Getting your own OpenAI API key

To use this app, you'll need to get your own [OpenAI](https://platform.openai.com) API key.

After signing up for an OpenAI account, you can access your API key from [this page](https://platform.openai.com/account/api-keys).
