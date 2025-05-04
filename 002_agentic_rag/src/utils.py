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
        
async def pull_ollama_model(model_name: str) -> bool:
    """
    Pulls the specified Ollama model if it's not already available.

    Args:
        model_name: The name of the Ollama model to pull (e.g., "llama3:latest").

    Returns:
        True if the model is available (either already present or successfully pulled),
        False otherwise.
    """
    print(f"Checking for model: {model_name}")
    try:
        # Check if the model exists locally
        process_check = await asyncio.create_subprocess_exec(
            "ollama", "list",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout_check, stderr_check = await process_check.communicate()

        if process_check.returncode != 0:
            print(f"Error checking for models: {stderr_check.decode()}", file=sys.stderr)
            return False

        if model_name in stdout_check.decode():
            print(f"Model '{model_name}' is already available.")
            return True

        # If the model is not listed, attempt to pull it
        print(f"Model '{model_name}' not found locally. Attempting to pull...")
        process_pull = await asyncio.create_subprocess_exec(
            "ollama", "pull", model_name,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Stream output for better user experience during pulling
        while True:
            line = await process_pull.stdout.readline()
            if not line:
                break
            print(line.decode().strip())

        await process_pull.wait() # Wait for the pull process to complete

        if process_pull.returncode == 0:
            print(f"Successfully pulled model: {model_name}")
            return True
        else:
            error_output = (await process_pull.stderr.read()).decode()
            print(f"Error pulling model {model_name}: {error_output}", file=sys.stderr)
            return False

    except FileNotFoundError:
        print("Error: 'ollama' command not found. Please ensure Ollama is installed and in your PATH.", file=sys.stderr)
        return False
    except Exception as e:
        print(f"An unexpected error occurred during model pulling: {e}", file=sys.stderr)
        return False