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

    # Satisfaction level
    if "average satisfaction" in q or "satisfaction level" in q:
        avg_satisfaction = df['satisfaction_level'].mean()
        return f"🧠 The average satisfaction level is {avg_satisfaction:.2f}."

    # Last evaluation
    elif "average evaluation" in q or "last evaluation" in q:
        avg_eval = df['last_evaluation'].mean()
        return f"🧠 The average last evaluation score is {avg_eval:.2f}."

    # Number of projects
    elif "number of projects" in q or "number project" in q:
        avg_projects = df['number_project'].mean()
        return f"🧠 The average number of projects per employee is {avg_projects:.2f}."

    # Monthly hours
    elif "average hours" in q or "monthly hours" in q:
        avg_hours = df['average_montly_hours'].mean()
        return f"🧠 The average monthly working hours are {avg_hours:.2f}."

    # Time spent in company
    elif "time in company" in q or "years in company" in q:
        avg_years = df['time_spend_company'].mean()
        return f"🧠 The average time spent in the company is {avg_years:.2f} years."

    # Work accident
    elif "work accident" in q:
        count_accident = df[df['Work_accident'] == 1].shape[0]
        return f"🧠 {count_accident} employees had work accidents."

    # Left / turnover
    elif "left" in q or "turnover" in q:
        count_left = df[df['left'] == 1].shape[0]
        return f"🧠 {count_left} employees left the company."

    # Promotions
    elif "promotion" in q or "promoted" in q:
        count_promo = df[df['promotion_last_5years'] == 1].shape[0]
        return f"🧠 {count_promo} employees got a promotion in the last 5 years."

    # Department
    elif "department" in q:
        depts = df['Department'].value_counts()
        return "🧠 Number of employees per department:\n" + "\n".join([f"{d}: {c}" for d, c in depts.items()])

    # Salary
    elif "salary" in q:
        # Count per category
        salary_counts = df['salary'].value_counts()
        # Average salary (map categories to numbers)
        mapping = {'low': 3000, 'medium': 5000, 'high': 7000}
        avg_salary = df['salary'].map(mapping).mean()
        response = "🧠 Salary Information:\n"
        response += "\n".join([f"{s}: {c} employees" for s, c in salary_counts.items()])
        response += f"\nAverage salary (estimated): {avg_salary:.2f}"
        return response

    # Fallback (TF-IDF search)
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
