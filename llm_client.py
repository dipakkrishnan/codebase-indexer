import openai
import anthropic
from phoenix.otel import register

tracer_provider = register(
  project_name="codebase-indexer",
  auto_instrument=True
)
tracer = tracer_provider.get_tracer(__name__)
import os
from typing import List, Optional, Union

class LLMClient:

    def __init__(self, provider: str = "openai"):
        self.provider = provider
        
        if provider == "openai":
            self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            self.default_embedding_model = "text-embedding-3-large"
            self.default_completion_model = "gpt-4o"
        elif provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            self.default_completion_model = "claude-3.7-sonnet"
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def generate_embeddings(self, text: str, model: Optional[str] = None) -> List[float]:
        model = model or self.default_embedding_model
        
        if self.provider == "openai":
            response = self.client.embeddings.create(
                model=model,
                input=text
            )
            return response.data[0].embedding
        else:
            raise ValueError("Embeddings can only be generated with OpenAI!")
    
    def generate_completion(self, prompt: str, model: Optional[str] = None) -> str:
        model = model or self.default_completion_model
        
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
