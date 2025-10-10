import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ----------------------------
# 💬 Choose which LLM to use
# ----------------------------
USE_LOCAL = False  # set True for local model via ollama, False for OpenAI

if USE_LOCAL:
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  # for local Llama3/Mistral
    MODEL_NAME = "llama3"
else:
    from openai import OpenAI
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])  # or os.environ["OPENAI_API_KEY"]
    MODEL_NAME = "gpt-4-turbo"

# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(page_title="🤖 Intelligent HR Q&A with LLM", page_icon="🧠", layout="wide")

st.markdown("""
<h1 style='text-align:center;color:#0a3d62;'>🤖 Intelligent HR Q&A System using RAG + LLM</h1>
<p style='text-align:center;'>Ask questions about your HR data — now powered by a real language model.</p>
""", unsafe_allow_html=True)

# ----------------------------
# Load Dataset
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("HR_comma_sep.csv")
    return df

df = load_data()

with st.expander("📊 View HR Dataset"):
    st.dataframe(df.head())

# ----------------------------
# Prepare Corpus (Retriever)
# ----------------------------
@st.cache_resource
def prepare_corpus():
    text_data = df.astype(str).apply(lambda x: ' '.join(x), axis=1).tolist()
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(text_data)
    return vectorizer, vectors, text_data

vectorizer, vectors, text_data = prepare_corpus()

# ----------------------------
# Retrieval + LLM Answer
# ----------------------------
def get_answer_llm(question):
    # --- Step 1: Retrieve ---
    q_vector = vectorizer.transform([question])
    similarity = cosine_similarity(q_vector, vectors).flatten()
    top_indices = similarity.argsort()[-3:][::-1]
    top_contexts = [text_data[i] for i in top_indices]

    context = "\n\n".join(top_contexts)

    # --- Step 2: Generate with LLM ---
    prompt = f"""
    You are an HR analytics assistant.
    Answer the user's question based on the HR dataset below.

    HR Data Context:
    {context}

    User Question:
    {question}

    Answer clearly and concisely in 2–3 sentences.
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": "You are an expert HR data analyst."},
                  {"role": "user", "content": prompt}],
        temperature=0.5,
    )

    return response.choices[0].message.content.strip()

# ----------------------------
# User Input
# ----------------------------
st.markdown("<h2>💬 Ask Your HR Question</h2>", unsafe_allow_html=True)
user_q = st.text_input("Enter your question:", placeholder="Example: What is the average satisfaction level of employees who left?")

if st.button("Get Answer"):
    if user_q.strip() == "":
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("🤖 Thinking..."):
            answer = get_answer_llm(user_q)
            st.success("✅ LLM Answer:")
            st.write(answer)

# ----------------------------
# Sidebar Info
# ----------------------------
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.write("""
    - Real **RAG system** using TF-IDF retrieval + LLM generation  
    - Works with OpenAI or local models (Llama3/Mistral)  
    - Ask complex HR questions — get natural answers  
    """)
    st.image("https://img.icons8.com/color/96/robot-2.png", width=80)
