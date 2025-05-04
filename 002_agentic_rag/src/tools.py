from llama_index.core import VectorStoreIndex
from llama_index.core.tools import FunctionTool
from llama_index.core.schema import Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec # Import DuckDuckGo tool spec
from huggingface_hub import list_models # Import list_models for Hugging Face tool
import random # Import random for weather tool

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

# Add DuckDuckGo Search Tool Class
class DDGSearchTool:
    def __init__(self):
        self.tool_spec = DuckDuckGoSearchToolSpec()
        self.tool = FunctionTool.from_defaults(self.tool_spec.duckduckgo_full_search)

    def search_tool(self, query: str) -> str:
        """Performs a full search using DuckDuckGo."""
        response = self.tool(query)
        # Assuming you want to return the body of the first result,
        # adjust this if you need different output handling
        if response and response.raw_output and response.raw_output[-1] and 'body' in response.raw_output[-1]:
             return response.raw_output[-1]['body']
        else:
             return "No search results found."

# Add Weather Info Tool Class
class WeatherInfoTool:
    def get_weather_info(self, location: str) -> str:
        """Fetches dummy weather information for a given location."""
        # Dummy weather data
        weather_conditions = [
            {"condition": "Rainy", "temp_c": 15},
            {"condition": "Clear", "temp_c": 25},
            {"condition": "Windy", "temp_c": 20}
        ]
        # Randomly select a weather condition
        data = random.choice(weather_conditions)
        return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"

# Add Hugging Face Hub Stats Tool Class
class HubStatsTool:
    def get_hub_stats(self, author: str) -> str:
        """Fetches the most downloaded model from a specific author on the Hugging Face Hub."""
        try:
            # List models from the specified author, sorted by downloads
            models = list(list_models(author=author, sort="downloads", direction=-1, limit=1))

            if models:
                model = models[0]
                return f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads."
            else:
                return f"No models found for author {author}."
        except Exception as e:
            return f"Error fetching models for {author}: {str(e)}"
            
