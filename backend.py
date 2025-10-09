# backend.py
import pandas as pd
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI

# 1. Load HR data
df = pd.read_csv("hr_data.csv")  # Make sure your CSV has a column named 'text'

# 2. Create embeddings
embeddings = OpenAIEmbeddings()

# 3. Build FAISS vector store for retrieval
vector_store = FAISS.from_texts(df['text'].tolist(), embeddings)

# 4. Create RAG (Retrieval-Augmented Generation) QA chain
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=vector_store.as_retriever()
)

# 5. Function to get answer from question
def get_answer(question):
    """
    Input: question (str)
    Output: answer (str) from RAG model
    """
    return qa.run(question)
