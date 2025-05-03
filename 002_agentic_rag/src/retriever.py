## Implements retrieval functions to support knowledge access

# src/agent.py

from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama
from llama_index.core.tools import FunctionTool
from src.tools import GuestInfoRetrieverTool
from src.utils import ensure_ollama_server
import traceback
from llama_index.core.schema import Document

async def query_guest_agent(
    docs: list[Document],
    question: str,
    tool_name: str = "guest_info",
    tool_description: str = "Retrieve comprehensive information about guests attending the gala by their name, including their relation, description, and contact details like email.",
    llm_model: str = "llama3:latest",
    system_prompt: str = "You are a helpful assistant providing information about guests. When asked about a guest, use the provided information to give a concise summary, including their name, relation, description, and email address if available."
):
    """
    Initializes the guest information agent with customizable parameters and queries it.

    Args:
        docs: A list of Document objects containing guest information.
        question: The question to ask the agent.
        tool_name: The name for the guest information tool.
        tool_description: The description for the guest information tool.
        llm_model: The name of the Ollama model to use.
        system_prompt: The system prompt for the agent.

    Returns:
        The response object from the agent.
    """
    # Initialize the guest info retriever tool
    guest_info_retriever = GuestInfoRetrieverTool(docs)

    # Create a proper FunctionTool using the provided arguments
    guest_info_tool = FunctionTool.from_defaults(
        fn=guest_info_retriever.get_guest_info,
        name=tool_name,
        description=tool_description
    )

    # Ensure Ollama server is running
    await ensure_ollama_server()

    # Initialize the LLM using the provided model name
    llm = Ollama(model=llm_model)

    # Create the agent with customizable parameters
    alfred = ReActAgent.from_tools(
        [guest_info_tool],
        llm=llm,
        verbose=False,  # Change to True for debugging if needed
        system_prompt=system_prompt
    )

    # Query the agent
    try:
        response = await alfred.aquery(question)
        return response
    except Exception as e:
        print(f"An error occurred during agent query: {e}")
        traceback.print_exc()
        return None