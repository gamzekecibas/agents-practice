import httpx
import subprocess
import asyncio

async def check_ollama_health():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags")
            response.raise_for_status()
            print("Ollama server is running and ready.")
            return True
    except httpx.RequestError as e:
        print(f"Ollama server is not reachable: {e}")
        return False

async def start_ollama_server():
    try:
        process = subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Ollama server is starting...")
        await asyncio.sleep(5)  # Wait for the server to start
        return process
    except Exception as e:
        print(f"Failed to start Ollama server: {e}")
        return None

async def ensure_ollama_server():
    if not await check_ollama_health():
        print("Attempting to start Ollama server...")
        process = await start_ollama_server()
        if process is None:
            raise RuntimeError("Failed to start Ollama server. Please start it manually.")
        print("Ollama server started successfully.")
    else:
        print("Ollama server is already running.")
