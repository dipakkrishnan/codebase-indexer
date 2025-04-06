import os
from glob import glob
from chunker import ChunkGenerator, SimpleChunkGenerator, ComplexChunkGenerator
import logging
from pymilvus import model
from db_client import get_db_client, get_embedder

logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log message format
    handlers=[
        logging.StreamHandler()  # Add a console handler
    ]
)

def find_python_files(directory: str) -> list[str]:
    """Find all Python files in a directory."""
    return glob(os.path.join(directory, '**', '*.py'), recursive=True)

def read_file(file_path: str) -> str:
    """Read the contents of a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

class Indexer: 

    def __init__(self, repo_path: str, chunk_generator: ChunkGenerator, repo_language: str = "python"):
        self.milvus_client = get_db_client()
        self.openai_embedder = get_embedder()
        if self.milvus_client.has_collection(collection_name="demo_collection"):
            self.milvus_client.drop_collection(collection_name="demo_collection")
        self.milvus_client.create_collection(
            collection_name="demo_collection",
            dimension=self.EMBEDDING_DIM,  # The vectors we will use in this demo has 768 dimensions
        )
        self.repo_path = repo_path
        self.repo_language = repo_language
        self.chunk_generator = chunk_generator
        self.process_source_code(self.repo_path)
            
    def process_source_code(self, repo_path: str) -> list:
        """Picks up source code from some directory on disk."""
        python_source_files = find_python_files(repo_path)
        self.chunk2file = {}
        self.source_code_documents = []
        for file in python_source_files:
            code_document = read_file(file)
            chunks = self.chunk_generator.generate(code_document, self.repo_language)
            for chunk in chunks:
                self.source_code_documents.append(chunk)
                self.chunk2file[chunk] = file

    def index(self):
        """Indexes source code into the backend db."""
        vectors = self.openai_embedder.encode_documents(self.source_code_documents)
        data = [
            {
                "id": i, 
                "vector": vectors[i], 
                "source_code": self.source_code_documents[i], 
                "codebase": self.repo_path.split("/")[-1],
                "source_file": self.chunk2file[self.source_code_documents[i]]
            }
            for i in range(len(vectors))
        ]
        self.milvus_client.insert(collection_name="demo_collection", data=data)
        logging.info(f"Indexed {len(self.source_code_documents)} documents!")


if __name__ == "__main__":
    if os.environ.get("chunking_strategy", "") == "simple":
        chunk_generator = SimpleChunkGenerator()
    else:
        chunk_generator = ComplexChunkGenerator()

    repo_path = os.environ["QA_REPO"]
    indexer = Indexer(repo_path, chunk_generator)
    indexer.index()
