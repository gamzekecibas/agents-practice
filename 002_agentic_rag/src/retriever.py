## Implements retrieval functions to support knowledge access

from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama
from llama_index.core.tools import FunctionTool
from src.tools import GuestInfoRetrieverTool
from src.utils import ensure_ollama_server
import traceback
from llama_index.core.schema import Document
import json

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
    
async def query_multi_step_agent(
    query: str,
    tools: dict[str, FunctionTool],
    llm_model: str = "llama3:latest",
    print_details: bool = False
):
    """
    Executes a multi-step agent workflow with planning, tool execution, and synthesis.

    Args:
        query: The user's question.
        llm: The Ollama LLM instance to use.
        tools: A dictionary mapping tool names to FunctionTool instances.
        print_details: Whether to print execution details (tool calls and results).

    Returns:
        The final synthesized answer from the agent.
    """
    # Step 1: Plan the execution
    planning_prompt = f"""
    Analyze the following user query and create a plan to answer it using the available tools.
    Carefully consider the description of each tool and choose the tool that is most relevant and specific to the task.
    Available tools:
    - dd_search_tool: Use to search the internet for general information.
    - hub_stats_tool: Use to find the most downloaded model for a Hugging Face author.
    - weather_info_tool: Use to get weather information for a location.

    Output the plan as a JSON array of steps. Each step should have:
    - "task": A description of the task for this step.
    - "tool": The name of the tool to use (from the available tools).
    - "tool_input": The input for the tool (e.g., a search query, an author name, a location).

    Example output format:
    ```json
    [
      {{
        "task": "Example task",
        "tool": "example_tool",
        "tool_input": {{"key": "value"}}
      }}
    ]
    ```

    User query: {query}
    """
    # Ensure Ollama server is running
    await ensure_ollama_server()

    # Initialize the LLM using the provided model name
    llm = Ollama(model=llm_model, request_timeout=1200)
    
    plan_response = await llm.acomplete(planning_prompt)

    # Extract the JSON plan from the LLM's output
    plan_text = plan_response.text.strip()
    json_start = plan_text.find("```json")
    json_end = plan_text.find("```", json_start + 7) # Find the closing ``` after the opening one

    plan = [] # Initialize plan as an empty list
    if json_start != -1 and json_end != -1:
        # Extract the text between the triple backticks
        json_string = plan_text[json_start + 7:json_end].strip()
        try:
            plan = json.loads(json_string)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON plan: {e}")
            print(f"Extracted JSON string: {json_string}")
            print("Error: Could not parse the planning response from the LLM.")
            return "Error: Could not parse the planning response from the LLM." # Return error message
    else:
        print("Error: Could not find JSON block in LLM output.")
        print(f"LLM output: {plan_text}")
        print("Error: Could not find the planning response in the expected format.")
        return "Error: Could not find the planning response in the expected format." # Return error message

    tool_results = {}
    
    # Step 2: Execute the plan
    if print_details:
        print("--- Execution Details ---")

    for step in plan:
        task = step["task"]
        tool_name = step["tool"]
        tool_input = step["tool_input"]

        if tool_name in tools:
            tool_instance = tools[tool_name]
            try:
                result = await tool_instance.acall(**tool_input) # Use **tool_input to unpack dict
                # Extract the text content from the ToolOutput object
                tool_results[task] = result.content
                if print_details:
                    print(f"Executed '{task}': {result.content}")
            except Exception as e:
                tool_results[task] = f"Error executing tool {tool_name}: {str(e)}\n" # Added newline for better formatting
                if print_details:
                    print(f"Error executing '{task}': {e}")
        else:
            tool_results[task] = f"Error: Tool '{tool_name}' not found."
            if print_details:
                print(f"Error: Tool '{tool_name}' not found.")

    if print_details:
        print("-----------------------")

    # Step 3: Synthesize the information
    synthesis_prompt = f"""
    The user asked: '{query}'.
    You gathered the following information:
    {json.dumps(tool_results, indent=2)}

    Please synthesize this information into a concise answer that directly addresses all parts of the original question.
    Avoid adding introductory or concluding remarks. Start directly with the answer.
    """
    
    synthesized_answer = await llm.acomplete(synthesis_prompt)

    return synthesized_answer.text