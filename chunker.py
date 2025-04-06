from typing import Any
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
from abc import abstractmethod, ABC
from code_parser import CodeParser

class ChunkGenerator(ABC):

    @abstractmethod
    def generate(self, *args: Any, **kwds: Any) -> Any:
        raise Exception("Cannot call generation directly on parent class.")


class SimpleChunkGenerator(ChunkGenerator):

    def generate(self, code_document: str, language: str) -> list[str]:
        """
        Generates a list of chunks from code document.
        A code document could be a singular file or a merge of many.

        :return: list of chunks split on language-specific separators.
        """
        lang_enum = Language[language.upper()]
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=lang_enum, chunk_size=50, chunk_overlap=0
        )
        documents = splitter.create_documents([code_document])
        return [doc.page_content for doc in documents]


class ComplexChunkGenerator:

    ts_parsers = {
        "python": CodeParser("python")   
    }

    def generate(self, code_document: str, language: str) -> list[str]:
        """
        Generates a list of chunks from code document.
        Chunks correspond to entire class definitions or standalone function definitions.

        :return: list of chunks.
        """
        parser = self.ts_parsers.get(language)
        if not parser:
            raise ValueError(f"Unsupported language: {language}")
            
        tree = parser.build_tree(code_document)
        chunks = []
        
        for node in tree.root_node.children:
            if node.type == "class_definition":
                class_chunk = code_document[node.start_byte:node.end_byte]
                chunks.append(class_chunk.strip())
            elif node.type == "function_definition":
                function_chunk = code_document[node.start_byte:node.end_byte]
                chunks.append(function_chunk.strip())
        return chunks
