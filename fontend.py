import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# LLM Setup
# ----------------------------
try:
    from openai import OpenAI
    client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", None))
    MODEL_NAME = "gpt-4-turbo"
    ONLINE = True
    if client.api_key is None:
        raise ValueError("No API key found")
except Exception:
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    MODEL_NAME = "llama3"
    ONLINE = False

# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(page_title="🤖 HR Q&A with RAG Fallback", page_icon="🧠", layout="wide")
st.markdown("""
<h1 style='text-align:center;color:#0a3d62;'>🤖 HR Q&A System with RAG Fallback</h1>
<p style='text-align:center;'>Ask questions — if LLM fails, top HR rows are shown instead.</p>
""", unsafe_allow_html=True)

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
def prepare_corpus():
    text_data = df.astype(str).apply(lambda x: ' '.join(x), axis=1).tolist()
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(text_data)
    return vectorizer, vectors, text_data

vectorizer, vectors, text_data = prepare_corpus()

# ----------------------------
# Store history
# ----------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

# ----------------------------
# RAG + LLM Answer Function
# ----------------------------
def get_answer_llm(question):
    # RAG retrieval
    q_vector = vectorizer.transform([question])
    similarity = cosine_similarity(q_vector, vectors).flatten()
    top_indices = similarity.argsort()[-3:][::-1]
    top_rows = df.iloc[top_indices]  # dataframe rows

    # Prepare prompt
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
        rag_fallback = None
    except Exception:
        answer = "⚠️ LLM failed; see top relevant HR rows below."
        rag_fallback = top_rows

    # Save to history
    st.session_state.history.append({
        "Question": question,
        "Answer": answer,
        "RAG Rows": rag_fallback
    })
    return answer, rag_fallback

# ----------------------------
# User Input
# ----------------------------
st.markdown("<h2>💬 Ask Your HR Question</h2>", unsafe_allow_html=True)
user_q = st.text_input("Enter your question:", placeholder="Example: What is the average satisfaction level?")

if st.button("Get Answer") and user_q.strip() != "":
    with st.spinner("🤖 Thinking..."):
        answer, rag_fallback = get_answer_llm(user_q)
        st.success("✅ Answer:")
        st.write(answer)
        if rag_fallback is not None:
