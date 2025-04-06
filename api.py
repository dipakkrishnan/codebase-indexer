from fastapi import FastAPI
from db_client import get_db_client, get_embedder
from llm_client import LLMClient
from models import PromptRequest
from prompts import ANSWER_PROMPT
import json

app = FastAPI()
db_client = get_db_client()
embedder = get_embedder()
llm_client = LLMClient()

@app.post("/answers")
async def generate_answers(request: PromptRequest):
    """POST method to generate answers from a user prompt."""
    query_vectors = embedder.encode_queries([request.prompt])
    ann_results = db_client.search(
        collection_name="demo_collection",
        data=query_vectors,
        limit=2,
        output_fields=["source_code", "codebase", "source_file"],
    )
    answer_prompt = ANSWER_PROMPT.format(
        code_context=json.dumps(ann_results),
        prompt = request.prompt
    )
    response = llm_client.generate_completion(answer_prompt)
    return {"answer": response}
