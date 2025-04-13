from llama_index.core import VectorStoreIndex
from llama_index.core.tools import FunctionTool
from llama_index.core.schema import Document
from llama_index.embeddings.ollama import OllamaEmbedding

class GuestInfoRetrieverTool:
    def __init__(self, docs, model_name="gemma2:2b"):
        # Use Ollama embedding model
        embed_model = OllamaEmbedding(model_name=model_name)
        
        # Initialize the index with the Ollama embedding model
        self.index = VectorStoreIndex.from_documents(
            docs, 
            embed_model=embed_model
        )
        # Get the retriever with hybrid search for better name matching
        self.retriever = self.index.as_retriever(
            similarity_top_k=3,  
        )

    def get_guest_info(self, query: str) -> str:
        """Get information about a guest by name or description"""
        # Use the retriever to find relevant documents
        nodes = self.retriever.retrieve(query)
        
        if not nodes:
            return f"No information found for '{query}'"
        
        # Look for exact matches in the retrieved nodes
        # This helps prioritize name matches over thematic matches
        for node in nodes:
            if query.lower() in node.text.lower():
                return node.text
        
        # If no exact match, return the top result
        return nodes[0].text