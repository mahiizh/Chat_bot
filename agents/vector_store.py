from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from .embeddings import get_embeddings
import os

class VectorStoreManager:
    def __init__(self, persist_directory="./chroma_db"):
        self.embeddings = get_embeddings()
        self.persist_directory = persist_directory
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def load_pdf(self, file_path):
        """Load and process PDF file"""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        self.vector_store.add_documents(chunks)
        return len(chunks)
    
    def load_text(self, file_path):
        """Load and process text file"""
        loader = TextLoader(file_path)
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        self.vector_store.add_documents(chunks)
        return len(chunks)
    
    def load_text_content(self, text_content, source_name="uploaded_content"):
        """Load text content directly (for OCR results)"""
        document = Document(
            page_content=text_content,
            metadata={"source": source_name}
        )
        chunks = self.text_splitter.split_documents([document])
        self.vector_store.add_documents(chunks)
        return len(chunks)
    
    def search(self, query, k=3):
        """Search for relevant documents"""
        results = self.vector_store.similarity_search(query, k=k)
        return results
    
    def get_retriever(self):
        """Get retriever for agent"""
        return self.vector_store.as_retriever(search_kwargs={"k": 3})