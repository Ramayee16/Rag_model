import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import traceback

# ----------------------------
# LLM Setup (Online / Offline)
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
st.set_page_config(page_title="🤖 Intelligent Q&A System using RAG", page_icon="🧠", layout="wide")
st.markdown("""
<h1 style='text-align:center;color:#0a3d62;'>🤖 Intelligent Question & Answer System using RAG Model</h1>
<p style='text-align:center;'>Ask any intelligent question — if the LLM is unavailable, RAG retrieval gives the best contextual answers.</p>
""", unsafe_allow_html=True)

# ----------------------------
# Load Dataset
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("HR_comma_sep.csv")

df = load_data()

with st.expander("📊 View Sample Dataset"):
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
# Store Q&A History
# ----------------------------
if "history" not in st.session_state:
    st.session_state["history"] = pd.DataFrame(columns=["Question", "Context (RAG)", "Answer"])

# ----------------------------
# Intelligent RAG + LLM Response
# ----------------------------
def get_answer_llm(question):
    # RAG retrieval
    q_vector = vectorizer.transform([question])
    similarity = cosine_similarity(q_vector, vectors).flatten()
    top_indices = similarity.argsort()[-3:][::-1]
    top_contexts = [text_data[i] for i in top_indices]
    context = "\n\n".join(top_contexts)

    # LLM prompt
    prompt = f"""
You are an intelligent data assistant specialized in analyzing HR and business datasets.
Use the following retrieved context to answer clearly and concisely (2–3 sentences).

Context from RAG:
{context}

User Question:
{question}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an intelligent assistant using RAG for precise answers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )
        answer = response.choices[0].message.content.strip()
    except Exception:
        answer = f"(LLM unavailable — showing top contextual insights)\n\n{context}"

    # Save to session history
    st.session_state.history.loc[len(st.session_state.history)] = [question, context, answer]
    return answer

# ----------------------------
# User Interaction
# ----------------------------
st.markdown("<h2>💬 Ask Your Intelligent Question</h2>", unsafe_allow_html=True)
user_q = st.text_input("Enter your question:", placeholder="Example: What is the average satisfaction level of employees?")

if st.button("Get Answer"):
    if user_q.strip() == "":
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("🤖 Thinking... please wait."):
            answer = get_answer_llm(user_q)
            st.success("✅ Answer:")
            st.write(answer)

# ----------------------------
# Show History
# ----------------------------
st.markdown("<h2>📝 Q&A History</h2>", unsafe_allow_html=True)
st.dataframe(st.session_state.history)
