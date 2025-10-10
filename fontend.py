import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Online OpenAI LLM Setup
# ----------------------------
from openai import OpenAI

try:
    # Try online OpenAI first
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    MODEL_NAME = "gpt-4-turbo"
    ONLINE = True
except KeyError:
    # Fallback to offline/local LLM
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    MODEL_NAME = "llama3"
    ONLINE = False


# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(page_title="🤖 HR Q&A Online", page_icon="🧠", layout="wide")

st.markdown("""
<h1 style='text-align:center;color:#0a3d62;'>🤖 HR Q&A System using RAG + GPT</h1>
<p style='text-align:center;'>Ask HR questions online and get answers powered by GPT.</p>
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
    q_vector = vectorizer.transform([question])
    similarity = cosine_similarity(q_vector, vectors).flatten()
    top_indices = similarity.argsort()[-3:][::-1]
    top_contexts = [text_data[i] for i in top_indices]
    context = "\n\n".join(top_contexts)

    prompt = f"""
    You are an HR analytics assistant.
    Use the HR dataset context to answer this question clearly in 2–3 sentences.

    HR Data Context:
    {context}

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
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ LLM Error: {str(e)}"

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
            st.success("✅ LLM Answer:")
            st.write(answer)

# ----------------------------
# Sidebar Info
# ----------------------------
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.write("""
    - Online RAG + GPT system  
    - Ask HR questions like satisfaction, promotions, salary, etc.  
    - Requires OpenAI API key stored in Streamlit Secrets
    """)
    st.image("https://img.icons8.com/color/96/robot-2.png", width=80)
