"""
RAG Pipeline with LangChain + Ollama
====================================

This script implements a Retrieval-Augmented Generation (RAG) pipeline
for contract clause question-answering using:
- FAISS vector search for clause retrieval
- Ollama for text generation
- LangChain for orchestration
- FastAPI for web interface

Usage:
    python ollama_qa.py

Dependencies:
    - langchain
    - langchain-community
    - fastapi
    - uvicorn
    - faiss-cpu
    - httpx (for Ollama API)
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# === 1. Paths ===
FAISS_INDEX_PATH = "../embeddings/output"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME = "gemma3"  # Change this if you're using another model

# === 2. Load Embeddings and FAISS Index ===
print("Loading FAISS index and embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, index_name="cuad_faiss", allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# === 3. Initialize Ollama LLM ===
print("Loading Ollama model...")
llm = Ollama(model=OLLAMA_MODEL_NAME)

# === 4. Prompt Template ===
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a legal contract assistant.\n\n"
        "Use the following contract clauses to answer the question as accurately as possible.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    )
)

# === 5. Create the RetrievalQA Chain ===
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
)



# === 6. Sample Query ===
if __name__ == "__main__":
    while True:
        query = input("\nAsk a legal question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        print("Querying...")
        result = qa_chain(query)

        print("\nAnswer:\n", result["result"])
        print("\nSources:")
        for doc in result["source_documents"]:
            print(" -", doc.metadata.get("source", "Unknown"))

def get_qa_chain():
    return qa_chain
