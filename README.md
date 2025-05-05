# AI Agents Practice Repository

A collection of projects exploring different aspects of AI agent development.

## Projects

### 1. 🤖 LLM Fundamentals Practice (`000_llm_practice`)
- Implementation of GPT from scratch using PyTorch
- Tokenization experiments with SentencePiece and tiktoken
- Dataset: Tiny Shakespeare

### 2. 🔍 Web Search Agent (`001_web_search_agent`)
- Terminal-based agent using LangGraph and DuckDuckGo Search
- Features tool usage and response formatting
- Implements a state machine for agent workflow

### 3. 📚 Agentic RAG System (`002_agentic_rag`)
- Retrieval Augmented Generation with LlamaIndex
- Custom retriever tool for guest information
- Ollama integration for local LLM operations
- For details, please check project [section](https://github.com/gamzekecibas/agents-practice/tree/main/002_agentic_rag) & Medium [post](https://medium.com/@gkecibas/building-an-agentic-rag-with-llamaindex-and-ollama-a-practical-guide-aa0fd9f43e41)!

## Setup

1. Clone repository:
```bash
git clone https://github.com/your-username/agents-practice.git
cd agents-practice
```

2. Create conda environment:
```bash
conda env create -f 000_llm_practice/001_gpt_from_scratch/agentenv_backup
conda activate agentenv
```
