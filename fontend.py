import streamlit as st
import pandas as pd
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(page_title="🤖 Intelligent Q&A System", page_icon="🧠", layout="wide")

st.title("🤖 Intelligent Question & Answer System using RAG")
st.write("Ask intelligent questions based on HR data. This system uses Retrieval-Augmented Generation (RAG) to give contextual answers.")

# ----------------------------
# Load Dataset
# ----------------------------
@st.cache_resource
def load_data():
    df = pd.read_csv("HR_comma_sep.csv")
    return df

df = load_data()

# Display dataset preview
with st.expander("📊 View HR Dataset"):
    st.dataframe(df.head())

# ----------------------------
# Build RAG Pipeline
# ----------------------------
@st.cache_resource
def build_rag_model():
    # Combine all text columns into one string per row
    text_data = df.astype(str).apply(lambda x: ' '.join(x), axis=1).tolist()

    # Split text into chunks
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.create_documents(text_data)

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Store in FAISS vector database
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Create retriever
    retriever = vectorstore.as_retriever()

    # Create LLM (GPT-like model)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Create RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa

qa_chain = build_rag_model()

# ----------------------------
# User Question Input
# ----------------------------
st.subheader("💬 Ask your question below:")
user_question = st.text_input("Enter your question:", placeholder="Example: What is the average monthly income of employees?")

if st.button("Get Answer"):
    if user_question.strip() == "":
        st.warning("⚠️ Please enter a valid question.")
    else:
        with st.spinner("🔍 Retrieving answer..."):
            response = qa_chain.invoke({"query": user_question})
            st.success("✅ Answer:")
            st.write(response["result"])

            # Show source data (optional)
            with st.expander("📚 View Retrieved Data"):
                for doc in response["source_documents"]:
                    st.write(doc.page_content)
