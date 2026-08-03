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