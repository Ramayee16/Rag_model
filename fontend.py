import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# ----------------------------
# OpenAI LLM Setup
# ----------------------------
# Make sure you add your OpenAI API key in Streamlit Secrets:
# OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxx"
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
MODEL_NAME = "gpt-4-turbo"

# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(page_title="🤖 HR Q&A with LLM", page_icon="🧠", layout="wide")

st.markdown("""
<h1 style='text-align:center;color:#0a3d62;'>🤖 Intelligent HR Q&A System using RAG + LLM</h1>
<p style='text-align:center;'>Ask questions about your HR data and get answers powered by GPT.</p>
""", unsafe_allow_html=True)

# ----------------------------
# Load HR Dataset
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("HR_comma_sep.csv")
    return df

df = load_data()

with st.expander("📊 View HR Dataset"):
    st.dataframe(df.head())

# ----------------------------
# Prepare TF-IDF Corpus (Retriever)
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
    # --- Step 1: Retrieve top 3 similar rows from HR data ---
    q_vector = vectorizer.transform([question])
    similarity = cosine_similarity(q_vector, vectors).flatten()
    top_indices = similarity.argsort()[-3:][::-1]
    top_contexts = [text_data[i] for i in top_indices]
    context = "\n\n".join(top_contexts)

    # --- Step 2: Generate answer using LLM ---
    prompt = f"""
    You are an HR analytics assistant.
    Use the following HR dataset context to answer the user's question.

    HR Data Context:
    {context}

    User Question:
    {question}

    Answer clearly and concisely in 2–3 sentences.
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful HR data analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return
