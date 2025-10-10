import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# ----------------------------
# ✅ Initialize OpenAI Client (for Streamlit Cloud)
# ----------------------------
# Make sure to add your OpenAI API key in Streamlit secrets:
# Go to Manage App → Settings → Secrets → add:
# OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxx"

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
MODEL_NAME = "gpt-4-turbo"

# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(page_title="🤖 Intelligent HR Q&A with LLM", page_icon="🧠", layout="wide")

st.markdown("""
<h1 style='text-align:center;color:#0a3d62;'>🤖 Intelligent HR Q&A System using RAG + LLM</h1>
<p style='text-align:center;'>Ask intelligent HR questions powered by GPT and offline HR data.</p>
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
# RAG + LLM Answer Function
# ----------------------------
def get_answer_llm(question):
    # Step 1: Retrieve similar HR data
    q_vector = vectorizer.transform([question])
    similarity = cosine_similarity(q_vector, vectors).flatten()
    top_indices = similarity.argsort()[-3:][::-1]
    top_contexts = [text_data[i] for i in top_indices]
    context = "\n\n".join(top_contexts)

    # Step 2: Generate LLM answer
    prompt = f"""
    You are an HR analytics assistant.
    Use the HR dataset context to answer this question.

    HR Data Context:
    {context}

    Question:
    {question}

    Answer concisely in 2–3 sentences.
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
        return f"⚠️ Error: {str(e)}"

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
    st.markdown("### ℹ️ About This App")
    st.write("""
    - Real **RAG system** using TF-IDF retrieval + GPT model  
    - Hosted online via **Streamlit Cloud**  
    - Requires your **OpenAI API key** in Streamlit Secrets  
    """)
    st.image("https://img.icons8.com/color/96/robot-2.png", width=80)
