import streamlit as st
from MLproject import get_answer  # import function from MLproject.ipynb

st.title("🤖 HR Intelligent Q&A System")
st.write("Ask questions about HR policies, leave, payroll, etc.")

question = st.text_input("Enter your question:")

if st.button("Submit"):
    answer = get_answer(question)
    st.write("Answer:", answer)
