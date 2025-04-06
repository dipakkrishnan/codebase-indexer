from pymilvus import MilvusClient
from pymilvus import model
import os

def get_db_client(db_path = "milvus_demo.db") -> MilvusClient:
    return MilvusClient(db_path)

def get_embedder(
        embedding_model: str = "text-embedding-3-large", 
        embedding_dim: int = 512
    ):
    return model.dense.OpenAIEmbeddingFunction(
        model_name=embedding_model,
        api_key=os.environ["OPENAI_API_KEY"],
        dimensions=embedding_dim
    )
