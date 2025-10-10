import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# ----------------------------
# 🔧 Local LLM Setup (Ollama)
# ----------------------------
from openai import OpenAI

client = OpenAI(api_key="your_openai_api_key_here")
MODEL_NAME = "gpt-3.5-turbo"

def get_llm_answer(question):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an intelligent assistant for HR data queries."},
            {"role": "user", "content": question},
        ],
        temperature=0.4,
    )
    return response.choices[0].message.content



# ----------------------------
# Streamlit UI Setup
# ----------------------------
st.set_page_config(page_title="🤖 Offline HR Q&A with LLM", page_icon="🧠", layout="wide")
st.markdown("<h1 style='text-align:center;color:#0a3d62;'>🤖 Intelligent HR Q&A System (Offline RAG)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Fully offline AI assistant using TF-IDF retrieval + local LLM via Ollama</p>", unsafe_allow_html=True)

# ----------------------------
# Load Dataset
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
def prepare_corpus(df):
    text_data = df.astype(str).apply(lambda x: " ".join(x), axis=1).tolist()
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(text_data)
    return vectorizer, vectors, text_data

vectorizer, vectors, text_data = prepare_corpus(df)

# ----------------------------
# Offline RAG Retrieval + Generation
# ----------------------------
def get_llm_answer(question):
    # Step 1 — Retrieve top HR rows
    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, vectors).flatten()
    top_idx = sims.argsort()[-3:][::-1]
    context = "\n\n".join(text_data[i] for i in top_idx)

    # Step 2 — Generate with local LLM
    prompt = f"""
    You are an HR analytics assistant.
    Use the HR dataset below to answer the user's question accurately.

    HR Data Context:
    {context}

    Question: {question}

    Provide a short, clear answer.
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful HR data assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )

    return response.choices[0].message.content.strip()

# ----------------------------
# Streamlit Input + Output
# ----------------------------
st.markdown("<h2>💬 Ask Your Question</h2>", unsafe_allow_html=True)
user_q = st.text_input("Type your question:", placeholder="Example: Which department has the highest average satisfaction?")

if st.button("Get Answer"):
    if user_q.strip():
        with st.spinner("🤖 Thinking using local LLM..."):
            answer = get_llm_answer(user_q)
        st.success("✅ Answer:")
        st.write(answer)
    else:
        st.warning("⚠️ Please enter a valid question.")

# ----------------------------
# Sidebar Info
# ----------------------------
with st.sidebar:
    st.markdown("### ℹ️ About this app")
    st.write("""
    - Runs **completely offline**  
    - Uses **TF-IDF** for retrieval and **Llama 3** for generation  
    - No API key, no internet required  
    """)
    st.image("https://img.icons8.com/color/96/robot-2.png", width=80)
