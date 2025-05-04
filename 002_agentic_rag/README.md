# Agentic RAG Use Case

This project is a playground for exploring Agentic Retrieval Augmented Generation (RAG) using LlamaIndex and Ollama. It was developed as part of Unit 3 ("Use Case for Agentic RAG") of the Hugging Face [Agent course](https://huggingface.co/learn/agents-course/unit3/agentic-rag/introduction). It demonstrates how to build an agent that can utilize multiple tools to answer complex queries by planning, executing tool calls, and synthesizing the results.

## Project Structure

- `agentic_rag.ipynb`: A Jupyter notebook demonstrating the individual components and how to combine them into an agent.
- `src/`: Contains the Python source code for the agent and its tools.
    - `app.py`: Integrates all components into a command-line application for running the agent.
    - `retriever.py`: Implements functions for querying the guest information agent and the multi-step agent.
    - `tools.py`: Defines the custom tools used by the agent, including a guest information retriever, DuckDuckGo search, weather information, and Hugging Face Hub stats.
    - `utils.py`: Contains utility functions for managing the Ollama server and models.
- `000_data_analysis/`: Contains for printing the small guest dataset from [agents-course/unit3-invitees](https://huggingface.co/datasets/agents-course/unit3-invitees).


## Features

- **Multi-tool Agent:** The agent can utilize different tools based on the user's query.
- **Planning and Execution:** The agent plans the steps required to answer a query, executes the necessary tool calls, and synthesizes the results.
- **Custom Tools:** Includes tools for:
    - Retrieving information about guests from a provided dataset.
    - Performing general web searches using DuckDuckGo.
    - Getting dummy weather information for a location.
    - Fetching statistics about models on the Hugging Face Hub.
- **Ollama Integration:** Uses Ollama for running local language models.
- **Interactive Command-Line Interface:** The `app.py` script provides a command-line interface to interact with the agent.

## Setup

1.  **Install Ollama:** If you don't have Ollama installed, follow the instructions on the [Ollama website](https://ollama.com/download).
2.  **Pull Required Models:** The project uses Ollama models. You will need to pull the models specified in the `agentic_rag.ipynb` notebook or when running `app.py`. For example, to pull the `llama3:latest` model, run:
    ```bash
    ollama pull llama3:latest
    ```
    The `app.py` script will attempt to pull the model if it's not found locally.
3.  **Create a Python Virtual Environment:** It's recommended to use a virtual environment.
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
4.  **Install Dependencies:** Install the required Python packages.
    ```bash
    pip install -r requirements.txt # You'll need to create this file
    ```
    *Note: A `requirements.txt` file was not provided, but you will need to create one containing the necessary libraries like `llama-index-core`, `llama-index-llms-ollama`, `llama-index-tools-duckduckgo`, `datasets`, `huggingface-hub`, `httpx`, `argparse`, etc.*

## Usage

### Running the Jupyter Notebook

Open the `agentic_rag.ipynb` notebook in a Jupyter environment (like JupyterLab or VS Code) and run the cells sequentially to see how the agent and tools work.

### Running the Command-Line Application

You can run the multi-tool agent from the command line using the `app.py` script.

```bash
python src/app.py <ollama_model_name> <user_query>
```

- `<ollama_model_name>`: The name of the Ollama model you want to use (e.g., `llama3:latest`, `gemma2:2b`).
- `<user_query>`: The question you want to ask the agent. If your query has spaces, enclose it in quotes.

**Examples:**

```bash
python src/app.py llama3:latest "What is Facebook and what's their most popular model?"
python src/app.py gemma2:2b "What's the weather like in Paris tonight?"
python src/app.py llama3:latest "Tell me about Lady Ada Lovelace."
```

The script will ensure the Ollama server is running and attempt to pull the specified model if it's not available. It will then process your query and print the agent's response.