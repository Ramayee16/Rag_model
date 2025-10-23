import streamlit as st
import tempfile
from backend_py import (
    extract_text_from_pdf, extract_text_from_docx, load_text_file,
    chunk_text, get_embedder, embed_texts, build_faiss_index, retrieve,
    generate_with_hf_improved, validate_hf_key
)

st.set_page_config(page_title="RAG Intelligent Document Q&A", layout="wide")
st.title("📄 Intelligent Document Q&A — Improved Retrieval")

# -----------------------------
# Session State
# -----------------------------
if 'hf_key' not in st.session_state:
    st.session_state['hf_key'] = None
if 'index' not in st.session_state:
    st.session_state['index'] = None
if 'chunks' not in st.session_state:
    st.session_state['chunks'] = []
if 'doc_uploaded' not in st.session_state:
    st.session_state['doc_uploaded'] = False

# -----------------------------
# Step 1: API Key
# -----------------------------
st.header("Step 1️⃣ — Enter Hugging Face API Key")
hf_key_input = st.text_input("Enter Hugging Face API Key:", type="password")

if st.button("Validate Key"):
    if hf_key_input.strip():
        if validate_hf_key(hf_key_input.strip()):
            st.session_state['hf_key'] = hf_key_input.strip()
            st.success("✅ API Key is valid! You can now upload documents.")
        else:
            st.session_state['hf_key'] = None
            st.error("❌ Invalid API Key!")
    else:
        st.warning("⚠️ Please enter your Hugging Face API key first.")

# -----------------------------
# Step 2: Upload Document
# -----------------------------
if st.session_state['hf_key']:
    st.header("Step 2️⃣ — Upload Document (PDF/DOCX/TXT)")
    uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx", "txt"])

    if uploaded_file and st.button("Process Document"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_file_path = tmp.name

        # Extract text
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(temp_file_path)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = extract_text_from_docx(temp_file_path)
        else:
            text = load_text_file(temp_file_path)

        if not text.strip():
            st.error("❌ No readable text found in this document.")
        else:
            st.info("🔍 Chunking and embedding the document...")
            chunks = chunk_text(text, chunk_size=600, overlap=100)
            embedder = get_embedder()
            embeddings = embed_texts(embedder, chunks)
            index = build_faiss_index(embeddings)

            st.session_state['index'] = index
            st.session_state['chunks'] = chunks
            st.session_state['doc_uploaded'] = True

            st.success(f"✅ Document processed successfully! ({len(chunks)} chunks created)")

# -----------------------------
# Step 3: Ask Question
# -----------------------------
if st.session_state['doc_uploaded']:
    st.header("Step 3️⃣ — Ask a Question About the Document")
    question = st.text_area("Enter your question here:")

    if st.button("Get Answer") and question.strip():
        with st.spinner("🤖 Retrieving answer from document..."):
            embedder = get_embedder()
            query_emb = embed_texts(embedder, [question])
            retrieved = retrieve(
                st.session_state['index'], query_emb, st.session_state['chunks'], k=10
            )
            contexts = [t[0] for t in retrieved if t[0].strip()]

            # Generate answer with highlighted snippet
            answer, source_chunk, method = generate_with_hf_improved(question, contexts)

            st.markdown(f"**Answer extraction method:** `{method}`")
            
            if source_chunk:
                # Highlight the answer in the chunk
                highlighted_chunk = source_chunk.replace(answer, f"**🟢 {answer} 🟢**")
                st.markdown("### 🔹 Chunk from which answer was extracted (preview with highlight):")
                st.write(highlighted_chunk[:800] + ("..." if len(highlighted_chunk) > 800 else ""))

            st.markdown("### 🧠 Answer:")
            st.write(answer)

else:
    if st.session_state['hf_key']:
        st.info("📥 Please upload a document first to start Q&A.")
    else:
        st.info("🔑 Please enter a valid Hugging Face API key first.")
