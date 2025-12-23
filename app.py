import os
import streamlit as st

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ================================
# LangSmith Configuration (Secrets)
# ================================
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]

# ================================
# Prompt Template
# ================================
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the question asked"),
        ("user", "Question: {question}")
    ]
)

# ================================
# Streamlit UI
# ================================
st.title("LangChain Demo With LLAMA3 Model")
input_text = st.text_input("What question do you have in mind?")

# ================================
# Ollama LLaMA3 Model
# ================================
llm = Ollama(model="llama3")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))
