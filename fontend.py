import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import traceback

# ----------------------------
# LLM Setup (Online or Local)
# ----------------------------
try:
    # Try online OpenAI first
    from openai import OpenAI
    client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", None))
    MODEL_NAME = "gpt-4-turbo"
    ONLINE = True
    if client.api_key is None:
        raise ValueError("No API key found")
except Exception:
    # Fallback to offline/local LLM
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    MODEL_NAME = "llama3"
    ONLINE = False

# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(page_title="🤖 HR Q&A", page_icon="🧠", layout="wide")
st.markdown("""
<h1 style='text-align:center;color:#0a3d62;'>🤖 HR Q&A System using RAG + LLM</h1>
<p style='text-align:center;'>Ask questions about HR data — online or offline LLM.</p>
""", unsafe_allow_html=True)

# ----------------------------
# Load HR Dataset
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("HR_comma_sep.csv")

df = load_data()
with st.expander("📊 View HR Dataset"):
    st.dataframe(df.head())

# ----------------------------
# Prepare TF-IDF Corpus
# ----------------------------
@st.cache_resource
def prepare_corpus():
    text_data = df.astype(str).apply(lambda x: ' '.join(x), axis=1).tolist()
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(text_data)
    return vectorizer, vectors, text_data

vectorizer, vectors, text_data = prepare_corpus()

# ----------------------------
# RAG + LLM Answer Function
# ----------------------------
def get_answer_llm(question):
    # Retrieve relevant HR rows
    q_vector = vectorizer.transform([question])
    similarity = cosine_similarity(q_vector, vectors).flatten()
    top_indices = similarity.argsort()[-3:][::-1]
    top_contexts = [text_data[i] for i in top_indices]
    context = "\n\n".join(top_contexts)

    prompt = f"""
You are an HR analytics assistant.
Answer the user's question based on the HR dataset context below in 2-3 sentences.

HR Data Context:
{context}

User Question:
{question}
"""

    # Try calling LLM
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful HR analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        # If LLM fails, fallback to just returning retrieved HR rows
        st.warning("⚠️ LLM call failed! Showing top HR rows instead.")
        st.text(traceback.format_exc())
        return f"{context}\n\n(This is retrieved HR data as fallback.)"

# ----------------------------
# User Input
# ---------------------------
