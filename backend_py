# -*- coding: utf-8 -*-
import os
import PyPDF2
import docx
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import re
from difflib import SequenceMatcher

# -----------------------------
# Text Extraction
# -----------------------------
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def load_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# -----------------------------
# Chunking
# -----------------------------
def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# -----------------------------
# Embeddings & FAISS
# -----------------------------
def get_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

def embed_texts(model, texts):
    return np.array(model.encode(texts, show_progress_bar=False, convert_to_numpy=True)).astype('float32')

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# -----------------------------
# Retrieval
# -----------------------------
def retrieve(index, query_emb, chunks, k=5):
    D, I = index.search(query_emb.astype('float32'), k)
    results = [(chunks[i], float(D[0][j])) for j, i in enumerate(I[0])]
    return results

# -----------------------------
# Hugging Face QA + Improved Answer Extraction
# -----------------------------
hf_qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

def _normalize(s: str) -> str:
    return re.sub(r'\s+', ' ', s.strip().lower())

def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def extract_answer_from_contexts_exact(contexts, question, window_chars=400):
    q_norm = _normalize(question)
    q_tokens = [t for t in re.split(r'[^a-z0-9]+', q_norm) if t]
    for ctx in contexts:
        ctx_low = ctx.lower()
        if q_norm in ctx_low:
            idx = ctx_low.find(q_norm)
            start = max(0, idx - window_chars//2)
            end = min(len(ctx), idx + len(q_norm) + window_chars//2)
            snippet = ctx[start:end].strip()
            return snippet, ctx
        if q_tokens:
            found = sum(1 for t in q_tokens if t in ctx_low)
            if found / max(1, len(q_tokens)) >= 0.5:
                for t in q_tokens:
                    if t in ctx_low:
                        idx = ctx_low.find(t)
                        start = max(0, idx - window_chars//2)
                        end = min(len(ctx), idx + window_chars//2)
                        return ctx[start:end].strip(), ctx
    return None, None

def extract_answer_from_contexts_fuzzy(contexts, question, threshold=0.6):
    q_norm = _normalize(question)
    best_ctx = None
    best_sim = 0.0
    for ctx in contexts:
        ctx_snippet = _normalize(ctx[:1000])
        sim = _similar(q_norm, ctx_snippet)
        if sim > best_sim:
            best_sim = sim
            best_ctx = ctx
    if best_sim >= threshold:
        return best_ctx[:400].strip(), best_ctx
    return None, None

def generate_with_hf_improved(question, contexts):
    snippet, src = extract_answer_from_contexts_exact(contexts, question)
    if snippet:
        return snippet, src, 'exact'

    snippet, src = extract_answer_from_contexts_fuzzy(contexts, question)
    if snippet:
        return snippet, src, 'fuzzy'

    best_answer = ""
    best_score = 0.0
    best_chunk = None
    for chunk in contexts:
        try:
            res = hf_qa_pipeline(question=question, context=chunk)
        except Exception:
            continue
        ans = res.get("answer", "").strip()
        score = res.get("score", 0)
        if ans and ans.lower() in chunk.lower() and score > best_score:
            best_answer = ans
            best_score = score
            best_chunk = chunk

    if best_answer:
        return best_answer, best_chunk, 'qa'

    return "No relevant content found in the uploaded document.", None, 'none'

# -----------------------------
# Hugging Face API Key Validation
# -----------------------------
def validate_hf_key(hf_key: str) -> bool:
    try:
        _ = pipeline("question-answering", model="deepset/roberta-base-squad2", use_auth_token=hf_key)
        return True
    except Exception:
        return False
