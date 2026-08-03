from typing import List
import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Model memory mein sirf EK BAAR load hoga (No rerun crashes)
@st.cache_resource(show_spinner="Loading Embedding Model...")
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

class LocalEmbeddings(Embeddings):
    def __init__(self):
        self.model = load_embedding_model()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, show_progress_bar=False).tolist()

class FAISSVectorStoreManager:
    def __init__(self):
        self.embeddings = LocalEmbeddings()
        self.vector_store = None

    def create_vector_store(self, chunks: List[Document]):
        if not chunks:
            return None
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        return self.vector_store

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        if not self.vector_store:
            return []
        return self.vector_store.similarity_search(query, k=k)