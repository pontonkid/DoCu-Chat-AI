import os
import fitz  # PyMuPDF for PDF processing
import faiss 
import numpy as np
import pickle
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq Client
client = Groq(api_key="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")

# Load Sentence Transformer model for embeddings
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Initialize FAISS index
INDEX_FILE = "faiss_index.pkl"

def load_faiss_index():
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "rb") as f:
            return pickle.load(f)
    return faiss.IndexFlatL2(384)

index = load_faiss_index()
documents = []

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(pdf_file)
    return "\n".join([page.get_text() for page in doc])

def chunk_text(text, chunk_size=500, overlap=100):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]

def add_to_faiss(text_chunks):
    global index, documents
    embeddings = embedding_model.encode(text_chunks)
    index.add(np.array(embeddings, dtype=np.float32))
    documents.extend(text_chunks)
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(index, f)

def query_faiss(query, top_k=3):
    query_embedding = embedding_model.encode([query])
    _, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
    return [documents[i] for i in indices[0] if i < len(documents)]

def query_groq(prompt):
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile"
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="RAG-based PDF Chatbot", page_icon="📄", layout="wide")

st.title("📄 RAG-based PDF Chatbot")
st.markdown("Talk to your PDFs using AI-powered search!")

with st.sidebar:
    st.subheader("📤 Upload a PDF")
    uploaded_file = st.file_uploader("Drag & drop or browse", type="pdf")

if uploaded_file:
    with st.spinner("Processing your PDF..."):
        with open("uploaded.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        text = extract_text_from_pdf("uploaded.pdf")
        text_chunks = chunk_text(text)
        add_to_faiss(text_chunks)
    
    st.sidebar.success("✅ PDF uploaded and indexed!")
    
    with st.expander("📃 Extracted Text Preview", expanded=False):
        st.text(text[:1000] + "...")
    
    st.markdown("---")
    st.subheader("🔍 Ask something about the document")
    query = st.text_input("Type your question below:")
    
    if query:
        retrieved_texts = query_faiss(query)
        
        if retrieved_texts:
            context = "\n".join(retrieved_texts)
            
            with st.expander("📖 Retrieved Context", expanded=False):
                st.text(context[:1000] + "...")
            
            response = query_groq(f"Context:\n{context}\n\nUser Query:\n{query}")
            
            st.subheader("💬 AI Response")
            st.markdown(f"**{response}**")
        else:
            st.warning("⚠️ No relevant context found in the document!")
