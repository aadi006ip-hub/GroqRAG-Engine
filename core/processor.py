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