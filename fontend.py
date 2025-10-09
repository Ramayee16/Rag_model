import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(page_title="🤖 Intelligent Q&A System", page_icon="🧠", layout="wide")

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>
body {
    background-color: #f0f2f6;
}
h1 {
    color: #0a3d62;
    text-align: center;
}
h2 {
    color: #1e3799;
}
.stButton>button {
    background-color: #1e3799;
    color: white;
    font-weight: bold;
}
.stTextInput>div>input {
    border-radius: 10px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Title Section
# ----------------------------
st.markdown("<h1>🤖 Intelligent HR Q&A System using RAG</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Ask intelligent questions about HR data. Offline version using TF-IDF retrieval.</p>", unsafe_allow_html=True)

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
# Prepare offline RAG system
# ----------------------------
@st.cache_resource
def prepare_corpus():
    text_data = df.astype(str).apply(lambda x: ' '.join(x), axis=1).tolist()
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(text_data)
    return vectorizer, vectors, text_data

vectorizer, vectors, text_data = prepare_corpus()

def get_answer_offline(question):
    q = question.lower()
    
    # Low salary
    if "low salary" in q:
        count = df[df['salary'] == 'low'].shape[0]
        return f"🧠 There are {count} employees with low salary."
    
    # High salary
    elif "high salary" in q:
        count = df[df['salary'] == 'high'].shape[0]
        return f"🧠 There are {count} employees with high salary."
    
    # Average salary
    elif "average salary" in q:
        mapping = {'low': 3000, 'medium': 5000, 'high': 7000}
        avg_salary = df['salary'].map(mapping).mean()
        return f"🧠 The average salary of employees is approximately {avg_salary:.2f}."
    
    # Turnover / left
    elif "how many people left" in q or "turnover" in q:
        count = df[df['left'] == 1].shape[0]
        return f"🧠 {count} employees have left the company."
    
    # Fallback: use TF-IDF similarity if question does not match keywords
    else:
        text_data = df.astype(str).apply(lambda x: ' '.join(x), axis=1).tolist()
        vectorizer = TfidfVectorizer(stop_words='english')
        vectors = vectorizer.fit_transform(text_data)
        q_vector = vectorizer.transform([question])
        similarity = cosine_similarity(q_vector, vectors).flatten()
        top_idx = np.argmax(similarity)
        best_match = text_data[top_idx]
        return f"🧠 Most relevant HR info:\n\n{best_match}"

# ----------------------------
# User Input (Centered Card)
# ----------------------------
st.markdown("<h2>💬 Ask Your Question</h2>", unsafe_allow_html=True)
user_question = st.text_input("Enter your question:", placeholder="Example: What is the average monthly salary?")

if st.button("Get Answer"):
    if user_question.strip() == "":
        st.warning("⚠️ Please enter a valid question.")
    else:
        with st.spinner("🔍 Finding the answer..."):
            answer = get_answer_offline(user_question)
            st.success("✅ Answer:")
            st.write(answer)

# ----------------------------
# Optional Sidebar with Info
# ----------------------------
with st.sidebar:
    st.markdown("<h2>ℹ️ About This App</h2>", unsafe_allow_html=True)
    st.write("""
    - Offline HR Q&A system using TF-IDF
    - Ask questions about employee satisfaction, salary, promotions, or departments
    - No API key required
    """)
    st.image("https://img.icons8.com/color/48/000000/robot-2.png", width=80)
