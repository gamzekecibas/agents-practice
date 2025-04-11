from llama_index.core import VectorStoreIndex # ✅ Import VectorStoreIndex
from llama_index.core.tools import FunctionTool
from llama_index.core.schema import Document
from llama_index.embeddings.ollama import OllamaEmbedding

class GuestInfoRetrieverTool:
    def __init__(self, docs):
        # Use Ollama embedding model
        embed_model = OllamaEmbedding(model_name="llama3:latest")  # Replace with your model
        
        # Initialize the index with the Ollama embedding model
        self.index = VectorStoreIndex.from_documents(
            docs, 
            embed_model=embed_model
        )
        # Get the default retriever (supports similarity search)
        self.retriever = self.index.as_retriever()

    def get_guest_info(self, query: str) -> str:
        # Use the retriever to find relevant documents
        nodes = self.retriever.retrieve(query)
        # Process nodes and return formatted info (implement your logic here)
        return f"Found {len(nodes)} results for '{query}'"