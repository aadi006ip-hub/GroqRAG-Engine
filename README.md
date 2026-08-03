# ⚡ GroqRAG Engine
GroqRAG Engine is a lightweight, ultra-fast Retrieval-Augmented Generation (RAG) system built with **LangChain**, **FAISS**, **HuggingFace Embeddings**, and **Groq LPU (`llama-3.3-70b-versatile`)**. It allows users to upload PDF, TXT, or Markdown documents and query them in natural language with sub-second response times and full source citation transparency.

---

## 🔗 Live Deployments & UI Links
✨ **Click on the badges below to interact with the project and view the user interface:**

[![Streamlit App](streamlit.jpg)](https://groqrag-engine-inrqdypw5loak5bkxmjdhp.streamlit.app/)

---

## 🌟 Key Features

* **⚡ Sub-Second Generation via Groq:** Utilizes Groq’s LPU hardware with `llama-3.3-70b-versatile` to deliver near-instant, context-grounded responses.
* **🔒 Zero-Cost Local Embeddings:** Embeds document text locally using `sentence-transformers/all-MiniLM-L6-v2`, eliminating dependency on paid third-party embedding APIs.
* **🔍 High-Performance Similarity Search:** Leverages an in-memory **FAISS** vector store to execute cosine similarity searches across document chunks.
* **🎨 Modular & Interactive Streamlit UI:** Provides an intuitive frontend to upload documents, adjust top-K retrieval parameters, chat with documents, and expand cited source chunks.

---

## 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **LLM Inference** | **Groq API** (`llama-3.3-70b-versatile`) | High-speed answer generation |
| **Vector Store** | **FAISS** (Facebook AI Similarity Search) | In-memory similarity search |
| **Embedding Model** | **HuggingFace** (`all-MiniLM-L6-v2`) | Local 384-dimensional dense vectors |
| **Orchestration** | **LangChain** | Pipeline chaining, prompt engineering, & parsing |
| **Frontend UI** | **Streamlit** | Dashboard interface with state management |
| **Document Parsing**| **PyPDF & LangChain Text Splitters** | Document extraction and recursive chunking |

---

## 📂 Project Directory Structure

```text
GroqRAG-Engine/
│
├── .env                 # Environment variables (Groq API Key)
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
│
├── core/
│   ├── __init__.py
│   ├── processor.py     # Step 1: Document Loading & Text Chunking
│   ├── vector_store.py  # Step 2 & 3: Local Embeddings & FAISS Indexing
│   └── rag_chain.py     # Step 4: Groq LLM Augmentation & Response Chain
│
└── app.py               # Streamlit Dashboard UI

```
## ⚙️ How It Works (End-to-End Pipeline)
```text
[ Document Input ] ──> [ Chunking ] ──> [ Local Embeddings ] ──> [ FAISS Vector Index ]
                                                                        │
[ User Query ] ──────> [ Vector Search ] ──> [ Top-K Chunks ] ──────────┤
                                                                        ▼
[ Streamlit UI ] <── [ Final Answer ] <── [ Groq API (Llama 3.3) ] <── [ Prompt Context ]

```
 1. **Document Ingestion & Chunking (processor.py):**
   Uploaded files (PDF, TXT, MD) are loaded and broken down into segments of 1,000 characters with a 200-character overlap using RecursiveCharacterTextSplitter.
 2. **Local Vector Embedding (vector_store.py):**
   Chunks are transformed into 384-dimensional dense vectors using HuggingFace's all-MiniLM-L6-v2 locally on your CPU—requiring no extra API keys.
 3. **FAISS Vector Indexing & Search (vector_store.py):**
   Vectors are indexed in an in-memory FAISS store. When a query is received, FAISS retrieves the K most semantically relevant document chunks.
 4. **Augmented Groq Generation (rag_chain.py):**
   The retrieved text chunks and user query are injected into a strict system prompt. Groq processes this context using llama-3.3-70b-versatile to produce an accurate, hallucination-free answer with cited sources.

---
## 🖥️ Application Dashboard

<div align="center">
  <img src="RAG UI.png" alt="GroqRAG Engine Web Application UI" width="90%" style="border-radius: 8px; border: 1px solid #ddd;"/>
  <br>
  <sup><i>Figure 1: Interactive Streamlit UI dashboard featuring GroqRAG Engine interface and dynamic filtering configurations.</i></sup>
</div>
---

## 🚀 Quickstart Guide
### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/GroqRAG-Engine.git](https://github.com/your-username/GroqRAG-Engine.git)
cd GroqRAG-Engine

```
### 2. Create & Activate a Virtual Environment
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate

```
### 3. Install Dependencies
```bash
pip install -r requirements.txt

```
### 4. Configure Environment Variables
Create a .env file in the project root directory:
```env
GROQ_API_KEY=your_groq_api_key_here

```
### 5. Run the Streamlit App
```bash
streamlit run app.py

```
## 📝 Complete Code Files
### requirements.txt
```text
streamlit>=1.35.0
langchain>=0.2.0
langchain-community>=0.2.0
langchain-groq>=0.1.0
langchain-huggingface>=0.0.1
sentence-transformers>=2.7.0
faiss-cpu>=1.8.0
pypdf>=4.2.0
python-dotenv>=1.0.1

```
### core/processor.py
```python
import os
import tempfile
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_and_chunk(self, uploaded_files) -> List[Document]:
        all_chunks = []
        for file in uploaded_files:
            ext = os.path.splitext(file.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_path = tmp_file.name

            try:
                if ext == ".pdf":
                    loader = PyPDFLoader(tmp_path)
                elif ext in [".txt", ".md"]:
                    loader = TextLoader(tmp_path, encoding="utf-8")
                else:
                    continue

                docs = loader.load()
                for d in docs:
                    d.metadata["source_name"] = file.name

                chunks = self.text_splitter.split_documents(docs)
                all_chunks.extend(chunks)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        return all_chunks

```
### core/vector_store.py
```python
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

```
### core/rag_chain.py
```python
from typing import List, Tuple
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

PROMPT_TEMPLATE = """You are a helpful research assistant.
Answer the question based ONLY on the provided document context below. 
If the answer cannot be deduced from the context, state clearly that you don't know based on the provided documents.

Context:
{context}

Question:
{question}
"""

class SimpleRAGChain:
    def __init__(self, groq_api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=0.2
        )
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    def answer(self, query: str, retrieved_docs: List[Document]) -> Tuple[str, List[Document]]:
        context_str = "\n\n---\n\n".join([
            f"[Source: {doc.metadata.get('source_name', 'Doc')}]\n{doc.page_content}"
            for doc in retrieved_docs
        ])

        chain = self.prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": context_str, "question": query})
        return response, retrieved_docs

```
### app.py
```python
import streamlit as st
import os
from core.processor import DocumentProcessor
from core.vector_store import FAISSVectorStoreManager
from core.rag_chain import SimpleRAGChain

st.set_page_config(page_title="GroqRAG Engine", page_icon="⚡", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_mgr" not in st.session_state:
    st.session_state.vector_mgr = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "indexed" not in st.session_state:
    st.session_state.indexed = False

with st.sidebar:
    st.title("⚡ GroqRAG Setup")
    
    groq_api_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
    selected_model = st.selectbox(
        "Groq Model", 
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
    )
    
    st.markdown("---")
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt", "md"], accept_multiple_files=True)
    
    k_chunks = st.slider("Chunks to retrieve (Top-K)", min_value=1, max_value=8, value=4)
    
    if st.button("🚀 Process & Index", use_container_width=True):
        if not groq_api_key:
            st.error("Please enter a Groq API Key.")
        elif not uploaded_files:
            st.warning("Please upload at least one file.")
        else:
            with st.spinner("Extracting, Chunking & Embedding..."):
                processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
                chunks = processor.load_and_chunk(uploaded_files)
                
                vector_mgr = FAISSVectorStoreManager()
                vector_mgr.create_vector_store(chunks)
                
                st.session_state.vector_mgr = vector_mgr
                st.session_state.rag_chain = SimpleRAGChain(groq_api_key=groq_api_key, model_name=selected_model)
                st.session_state.indexed = True
                
                st.success(f"Indexed {len(chunks)} chunks successfully!")

st.title("⚡ GroqRAG Engine")
st.caption("Sub-Second PDF Q&A Engine using HuggingFace Local Embeddings, FAISS, and Groq LPU")

if st.session_state.indexed:
    st.info("✅ Knowledge base indexed and ready for queries!", icon="💡")
else:
    st.warning("👈 Enter your Groq API Key and upload files in the sidebar to begin.", icon="⚠️")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 View Cited Sources"):
                for idx, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Chunk {idx}** from `{src.metadata.get('source_name', 'Doc')}`")
                    st.caption(src.page_content)

if prompt := st.chat_input("Ask a question about your uploaded document..."):
    if not st.session_state.indexed:
        st.error("Please index documents first!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching FAISS & generating answer with Groq..."):
                docs = st.session_state.vector_mgr.similarity_search(prompt, k=k_chunks)
                response, sources = st.session_state.rag_chain.answer(prompt, docs)
                
                st.write(response)
                if sources:
                    with st.expander("🔍 View Cited Sources"):
                        for idx, src in enumerate(sources, 1):
                            st.markdown(f"**Chunk {idx}** from `{src.metadata.get('source_name', 'Doc')}`")
                            st.caption(src.page_content)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources
                })

```
## 📄 License
Distributed under the MIT License. See LICENSE for more details.
```

```
