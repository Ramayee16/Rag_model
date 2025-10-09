import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(page_title="🤖 Intelligent Q&A System", page_icon="🧠", layout="wide")

st.title("🤖 Intelligent Question & Answer System using RAG (Offline Mode)")
st.write("Ask intelligent questions based on HR data — this version works offline using text similarity (no API key needed).")

# ----------------------------
# Load Dataset
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("HR_comma_sep.csv")
    return df

df = load_data()

# Display dataset
with st.expander("📊 View HR Dataset"):
    st.dataframe(df.head())

# ----------------------------
# Prepare text data for retrieval
# ----------------------------
@st.cache_resource
def prepare_corpus():
    # Combine all text columns into one string per row
    text_data = df.astype(str).apply(lambda x: ' '.join(x), axis=1).tolist()

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(text_data)
    return vectorizer, vectors, text_data

vectorizer, vectors, text_data = prepare_corpus()

# ----------------------------
# Offline QA System
# ----------------------------
def get_answer_offline(question):
    # Convert question to vector
    q_vector = vectorizer.transform([question])

    # Compute cosine similarity between question and text chunks
    similarity = cosine_similarity(q_vector, vectors).flatten()

    # Get most similar record
    top_idx = np.argmax(similarity)
    best_match = text_data[top_idx]
    score = similarity[top_idx]

    if score < 0.05:
        return "❌ Sorry, I couldn't find any relevant information in the HR data."
    else:
        return f"🧠 Based on HR data, here's the most relevant information:\n\n{best_match}"

# ----------------------------
# User Input
# ----------------------------
st.subheader("💬 Ask your question below:")
user_question = st.text_input("Enter your question:", placeholder="Example: What is the average monthly income of employees?")

if st.button("Get Answer"):
    if user_question.strip() == "":
        st.warning("⚠️ Please enter a valid question.")
    else:
        with st.spinner("🔍 Retrieving answer from HR database..."):
            answer = get_answer_offline(user_question)
            st.success("✅ Answer:")
            st.write(answer)
