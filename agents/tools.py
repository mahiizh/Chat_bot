from langchain_core.tools import tool
from tavily import TavilyClient
import os


@tool
def calculator(expression: str) -> str:
    """Use this for mathematical calculations."""
    try:
        return str(eval(expression))
    except Exception:
        return "Invalid math expression."


@tool
def web_search(query: str) -> str:
    """Search the web for real-time or current information."""
    api_key = os.getenv("TAVILY_API_KEY")
    
    if not api_key:
        return "TAVILY_API_KEY is not set."
    
    tavily = TavilyClient(api_key=api_key)
    results = tavily.search(query=query, max_results=3)
    
    summaries = [r["content"] for r in results["results"]]
    return "\n\n".join(summaries)


@tool
def search_documents(query: str) -> str:
    """Search through uploaded documents and images to find relevant information."""
    from .vector_store import VectorStoreManager
    
    try:
        vector_store = VectorStoreManager()
        results = vector_store.search(query, k=3)
        
        if not results:
            return "No relevant documents found. Please upload documents first."
        
        context = "\n\n".join([doc.page_content for doc in results])
        return f"Found relevant information:\n\n{context}"
    except Exception as e:
        return f"Error searching documents: {str(e)}"


def get_tools():
    return [calculator, web_search, search_documents]