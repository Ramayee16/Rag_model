import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import traceback

# ----------------------------
# LLM Setup (Online or Local)
# ----------------------------
from openai import OpenAI

try:
    # Try online OpenAI first
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    MODEL_NAME = "gpt-4-turbo"
    ONLINE = True
except KeyError:
    # Fallback to local LLM (Llama3/Mistral)
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    MODEL_NAME = "llama3"
    ONLINE = False

# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(page_title="🤖 HR Q&A System", page_icon="🧠", layout="wide")

st.markdown("""
<h1 style='text-align:center;color:#0a3d62;'>🤖 HR Q&A System using RAG + LLM</h1>
<p style='text-align:center;'>Ask HR questions and get answers powered by GPT or fallback RAG.</p>
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
# Prepare TF-IDF Corpus for RAG
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
    # RAG retrieval
    q_vector = vectorizer.transform([question])
    similarity = cosine_similarity(q_vector, vectors).flatten()
    top_indices = similarity.argsort()[-3:][::-1]
    top_rows = df.iloc[top_indices]  # get actual dataframe rows

    # Prepare LLM prompt
    context_text = top_rows.astype(str).apply(lambda x: ' '.join(x), axis=1).tolist()
    prompt = f"""
You are an HR analytics assistant.
Answer the user's question using the HR dataset context in 2-3 sentences.

HR Data Context:
{'\n'.join(context_text)}

User Question:
{question}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful HR analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )
        answer = response.choices[0].message.content.strip()
    except Exception:
        # Fallback: show top rows as a clean table
        st.warning("(LLM failed; showing top HR rows)")
        st.dataframe(top_rows)
        answer = "See the table above for the most relevant HR rows."

    return answer

# ----------------------------
# User Input
# ----------------------------
st.markdown("<h2>💬 Ask Your HR Question</h2>", unsafe_allow_html=True)
user_q = st.text_input("Enter your question:", placeholder="Example: What is the average satisfaction level?")

if st.button("Get Answer"):
    if user_q.strip() == "":
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("🤖 Thinking..."):
            answer = get_answer_llm(user_q)
            st.success("✅ Answer:")
            st.write(answer)

# ----------------------------
# Sidebar Info
# ----------------------------
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.write("""
    - RAG + LLM system for HR questions  
    - Works with OpenAI GPT or local LLM (like Llama3/Mistral)  
    - Shows fallback RAG table if LLM fails  
    """)
    st.image("https://img.icons8.com/color/96/robot-2.png", width=80)
