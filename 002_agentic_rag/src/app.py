## Integrates all components into a fully functional agent, which we’ll finalize in the last part of this unit
import asyncio
import sys
import json
import re
import argparse # Import argparse
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama
from llama_index.core.agent import ReActAgent
from llama_index.core.schema import Document
from tools import DDGSearchTool, WeatherInfoTool, HubStatsTool, GuestInfoRetrieverTool
from utils import ensure_ollama_server, pull_ollama_model
import datasets

# --- Tool Initialization ---
# Load the guest dataset (assuming it's available)
try:
    guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")
    docs = [
        Document(
            text="\n".join([
                f"Name: {guest_dataset['name'][i]}",
                f"Relation: {guest_dataset['relation'][i]}",
                f"Description: {guest_dataset['description'][i]}",
                f"Email: {guest_dataset['email'][i]}"
            ]),
            metadata={"name": guest_dataset['name'][i]}
        )
        for i in range(len(guest_dataset))
    ]
    gitr = GuestInfoRetrieverTool(docs)
    guest_info_tool = FunctionTool.from_defaults(
        fn=gitr.get_guest_info,
        name="guest_info_tool",
        description="Retrieve comprehensive information about guests attending the gala by their name, including their relation, description, and contact details like email."
    )
except Exception as e:
    print(f"Warning: Could not load guest dataset or initialize guest_info_tool: {e}", file=sys.stderr)
    guest_info_tool = None # Set to None if initialization fails

# Initialize other tools
ddst = DDGSearchTool()
search_tool = FunctionTool.from_defaults(
    fn=ddst.search_tool,
    name="dd_search_tool",
    description=(
        "Use this tool to search the internet for general information about a topic, "
        "definitions, explanations, or factual details that are likely to be found on websites. "
        "This is a general-purpose search tool."
    )
)

wit = WeatherInfoTool()
weather_info_tool = FunctionTool.from_defaults(
    fn=wit.get_weather_info,
    name="weather_info_tool",
    description="Use this tool ONLY when the user is asking for weather information for a specific location."
)

hst = HubStatsTool()
hub_stats_tool = FunctionTool.from_defaults(
    fn=hst.get_hub_stats,
    name="hub_stats_tool",
    description=(
        "Use this tool *only* to find statistics about models on the Hugging Face Hub. "
        "Specifically, use this tool to find the most downloaded model for a given author or organization *registered on the Hugging Face Hub*. "
        "Input should be the exact author or organization name (e.g., 'facebook', 'google', 'microsoft'). "
        "Example queries this tool can answer: 'What is the most downloaded model by google on Hugging Face?', 'Tell me about the stats for microsoft on the Hub.'"
    )
)

# Collect all available tools
available_tools = [search_tool, weather_info_tool, hub_stats_tool]
if guest_info_tool: # Only add guest_info_tool if it was successfully initialized
    available_tools.append(guest_info_tool)

# --- Agent Function ---
async def run_interactive_agent(query: str, llm_model: str):
    """
    Runs the multi-tool agent with the given query and LLM model.
    Attempts to pull the model if not available.
    """
    print(f"Using model: {llm_model}")

    # Ensure Ollama server is running and pull the model if needed
    await ensure_ollama_server()
    if not await pull_ollama_model(llm_model):
        print(f"Failed to pull model: {llm_model}. Exiting.", file=sys.stderr)
        return

    # Initialize the LLM
    llm = Ollama(model=llm_model, request_timeout=1200)

    # Create the agent
    agent = ReActAgent.from_tools(
        available_tools,
        llm=llm,
        verbose=True, # Set to True to see the agent's thought process
        system_prompt=(
                    "You are a helpful AI assistant that can use various tools to answer questions. "
                    "When you have gathered the necessary information using your tools, "
                    "synthesize the observations into a clear and concise answer that directly addresses the user's query. "
                    "Do not just repeat the tool output; rephrase it naturally."
                     )
                )

    print(f"\nProcessing query: '{query}'...")
    try:
        response = await agent.aquery(query)
        print("\n🎩 Agent's Response:")
        print(response)
    except Exception as e:
        print(f"\nAn error occurred during agent query: {e}", file=sys.stderr)
        # traceback.print_exc() # Uncomment for detailed traceback

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a multi-tool agent with a specified Ollama model and query.")
    parser.add_argument(
        "ollama_model_name",
        type=str,
        help="The name of the Ollama model to use (e.g., llama3:latest)"
    )
    parser.add_argument(
        "user_query",
        type=str,
        nargs='+', # This allows multiple words for the query
        help="The query you want the agent to process"
    )

    args = parser.parse_args()

    llm_model_name = args.ollama_model_name
    user_query = " ".join(args.user_query) # Join the list of words back into a single query string

    asyncio.run(run_interactive_agent(user_query, llm_model_name))