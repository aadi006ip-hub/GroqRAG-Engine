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