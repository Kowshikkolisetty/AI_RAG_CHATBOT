import os
import unittest

from src.graph import build_rag_graph


class TestRagGraph(unittest.TestCase):
    def test_build_rag_graph_without_api_keys(self):
        original_openai = os.environ.get("OPENAI_API_KEY")
        original_pinecone = os.environ.get("PINECONE_API_KEY")
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("PINECONE_API_KEY", None)

        try:
            graph = build_rag_graph(index_name="test-index")
            self.assertIsNotNone(graph)
            result = graph.invoke({"question": "What is Agentic AI?", "context": [], "answer": "", "score": 0.0})
            self.assertIn("answer", result)
        finally:
            if original_openai is not None:
                os.environ["OPENAI_API_KEY"] = original_openai
            if original_pinecone is not None:
                os.environ["PINECONE_API_KEY"] = original_pinecone


if __name__ == "__main__":
    unittest.main()
