import requests
import os

server_url = os.environ.get("CB_INDEXER_URL") or "http://localhost:8000/answers"

def qa(prompt: str):
    """Prompt the code-based RAG system to get an answer."""
    response = requests.post(server_url, json={"prompt": prompt})
    response.raise_for_status()
    return response.json()

resp = qa("what does this codebase do?")
print(resp)
