import ollama
import json
from typing import List, Dict, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

class AgentRes(TypedDict):
    tool_name: str
    tool_input: dict
    tool_output: str | None

class AgentWebSearch:
    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name
        self._initialize_tools()
        self._initialize_prompt()
        self._compile_workflow()
        self._check_model_availability()

    def _check_model_availability(self):
        """Ensure the selected model is available"""
        try:
            ollama.show(self.model_name)
        except (ollama.ResponseError, ConnectionError):
            print(f"Pulling {self.model_name} model...")
            ollama.pull(self.model_name)
            print("Model pulled successfully!")

    def _initialize_tools(self):
        """Initialize tools and tool descriptions"""
        self.tools = {
            "tool_browser": self._create_tool_browser(),
            "final_answer": self._create_final_answer()
        }
        self.tool_descriptions = "\n".join(
            f"{i+1}. `{name}`: {tool.description}"
            for i, (name, tool) in enumerate(self.tools.items())
        )

    def _create_tool_browser(self):
        """Create web search tool"""
        search = DuckDuckGoSearchRun()
        @tool
        def tool_browser(query: str) -> str:
            """Search the web for current events and factual information"""
            return search.run(query)
        return tool_browser

    def _create_final_answer(self):
        """Create final answer tool"""
        @tool
        def final_answer(answer: str) -> str:
            """Return final answer to user question"""
            return answer
        return final_answer

    def _initialize_prompt(self):
        """Initialize the prompt for the agent"""
        self.prompt = """
                You know everything, you must answer every question from the user, you can use the list of tools provided to you.
                Your goal is to provide the user with the best possible answer, including key information about the sources and tools used.

                Note, when using a tool, you provide the tool name and the arguments to use in JSON format. 
                For each call, you MUST ONLY use one tool AND the response format must ALWAYS be in the pattern:
                ```json
                {"name":"<tool_name>", "parameters": {"<tool_input_key>":<tool_input_value>}}
                ```
                Remember, do NOT use any tool with the same query more than once.
                Remember, if the user doesn't ask a specific question, you MUST use the `final_answer` tool directly.

                Every time the user asks a question, you take note of some keywords in the memory.
                Every time you find some information related to the user's question, you take note of some keywords in the memory.

                You should aim to collect information from a diverse range of sources before providing the answer to the user. 
                Once you have collected plenty of information to answer the user's question use the `final_answer` tool.
                """

    def _compile_workflow(self):
        """Compile the workflow for the agent
        Define the state schema or input and output parameters"""
        state_schema = {
            "input": str,
            "output": Dict
        }
        self.workflow = StateGraph(state_schema=state_schema)
        self.workflow.add_node(node="Agent", action=self._node_agent)
        self.workflow.set_entry_point(key="Agent")
        for k in self.tools.keys():
            self.workflow.add_node(node=k, action=self._node_tool)
        self.workflow.add_conditional_edges(source="Agent", path=self._conditional_edges)
        for k in self.tools.keys():
            if k != "final_answer":
                self.workflow.add_edge(start_key=k, end_key="Agent")
        self.workflow.add_edge(start_key="final_answer", end_key=END)


    def _node_agent(self, state):
        """Node for the agent"""
        query = state['input']
        messages = [
            {"role": "system", "content": self.prompt + "\n" + self.tool_descriptions},
            {"role": "user", "content": query}
        ]
        response = ollama.chat(
            model=self.model_name,
            messages=messages,
            format="json"
        )
        try:
            tool_call = json.loads(response['message']['content'])
        except json.JSONDecodeError:
            print("Error decoding JSON response from the model.")
            tool_call = {"name": "final_answer", "parameters": {"answer": response['message']['content']}}
        state['output'] = tool_call
        return state

    def _node_tool(self, state):
        """Node for the tool"""
        tool_call = state['output']
        tool_name = tool_call['name']
        tool_input = tool_call['parameters']
        if tool_name in self.tools:
            tool_output = self.tools[tool_name](**tool_input)
            state['output']['tool_output'] = tool_output
        else:
            state['output']['tool_output'] = "Tool not found."
        return state

    def _conditional_edges(self, state):
        """Conditional edges for the workflow"""
        tool_call = state['output']
        tool_name = tool_call['name']
        if tool_name == "final_answer":
            return "final_answer"
        else:
            return tool_name

    def interact(self):
        """Interact with the agent in the terminal"""
        print("Welcome to the TerminalAgent. Type 'exit' to quit.")
        while True:
            query = input("User: ")
            if query.lower() == 'exit':
                print("Goodbye!")
                break
            state = {"input": query}
            out = self.workflow.run(state)
            print(f"Agent: {out['output']['tool_output']}")