import tree_sitter_python as tspython
from typing import Generator
from tree_sitter import Language, Parser, Tree, Node


class CodeParser:

    lang2ts = {
        "python": Language(tspython.language())
    }

    def __init__(self, language: str):
        self.parser = Parser(self.lang2ts[language])

    def build_tree(self, code_document: str) -> Tree:
        """Builds a concrete parse tree representation of given code."""
        return self.parser.parse(bytes(code_document, "utf-8"))

    def traverse_tree(self, tree: Tree) -> Generator[Node, None, None]:
        """Traverses a node and its children in depth-first manner."""
        cursor = tree.walk()
        visited_children = False
        while True:
            if not visited_children:
                yield cursor.node
                if not cursor.goto_first_child():
                    visited_children = True
            elif cursor.goto_next_sibling():
                visited_children = False
            elif not cursor.goto_parent():
                break
