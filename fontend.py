import streamlit as st
import pandas as pd
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA


# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(page_title="🤖 Intelligent Q&A System", page_icon="🧠", layout="wide")

st.title("🤖 Intelligent Question & Answer System using RAG")
st.write("Ask questions related to HR data — this app retrieves and generates intelligent answers using the RAG model.")

# ----------------------------
# Load Dataset
# ----------------------------
@st.cache_resource
def load_data():
    df = pd.read_csv("HR_comma_sep.csv")
    return df

df = load_data()

# ----------------------------
# Prepare RAG Model
# ----------------------------
@st.cache_resource
def build_rag():
    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Convert text column to list (adjust column name if needed)
    text
