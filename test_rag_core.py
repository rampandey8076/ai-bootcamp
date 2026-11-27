import tempfile
import unittest
from pathlib import Path

import rag_core


class TestRAGCore(unittest.TestCase):

    def test_text_extraction(self):
        content = b"First paragraph.\nSecond paragraph."
        pages = rag_core.extract_text_pages(content)

        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0]["page"], 1)
        self.assertIn("First paragraph", pages[0]["text"])

    def test_text_extraction_with_page_break(self):
        content = b"Page one.\fPage two."
        pages = rag_core.extract_text_pages(content)

        self.assertEqual(len(pages), 2)
        self.assertEqual(pages[0]["page"], 1)
        self.assertEqual(pages[1]["page"], 2)

    def test_chunking(self):
        documents = [{
            "source": "sample.md",
            "page": 1,
            "text": " ".join(["embedding"] * 400)
        }]

        chunks = rag_core.split_into_chunks(documents)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0]["source"], "sample.md")
        self.assertEqual(chunks[0]["page"], 1)

    def test_chunk_overlap(self):
        words = [f"word{i}" for i in range(400)]

        documents = [{
            "source": "sample.md",
            "page": 1,
            "text": " ".join(words)
        }]

        chunks = rag_core.split_into_chunks(documents)

        first_words = chunks[0]["text"].split()
        second_words = chunks[1]["text"].split()

        self.assertEqual(
            first_words[-rag_core.CHUNK_OVERLAP:],
            second_words[:rag_core.CHUNK_OVERLAP]
        )

    def test_empty_document(self):
        documents = [{
            "source": "empty.txt",
            "page": 1,
            "text": "   \n   "
        }]

        chunks = rag_core.split_into_chunks(documents)

        self.assertEqual(chunks, [])

    def test_sample_documents_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)

            (path / "sample.md").write_text(
                "# Embeddings\nText embeddings represent meaning.",
                encoding="utf-8"
            )

            (path / "sample.txt").write_text(
                "Vector databases store vectors.",
                encoding="utf-8"
            )

            (path / "ignored.jpg").write_bytes(b"image")

            documents = rag_core.load_sample_documents(folder)

            self.assertEqual(len(documents), 2)

            filenames = {doc["source"] for doc in documents}

            self.assertIn("sample.md", filenames)
            self.assertIn("sample.txt", filenames)
            self.assertNotIn("ignored.jpg", filenames)

    def test_client_rejects_empty_api_key(self):
        with self.assertRaises(ValueError):
            rag_core.create_client("")

    def test_client_rejects_whitespace_api_key(self):
        with self.assertRaises(ValueError):
            rag_core.create_client("   ")


    def test_retrieve_filters_irrelevant_chunks(self):
        import numpy as np
        from types import SimpleNamespace

        class FakeClient:
            class Models:
                def embed_content(self, **kwargs):
                    embedding = SimpleNamespace(
                        values=[1.0, 0.0]
                    )
                    return SimpleNamespace(
                        embeddings=[embedding]
                    )

            models = Models()

        index = {
            "chunks": [
                {"text": "Relevant", "source": "a.md", "page": 1},
                {"text": "Irrelevant", "source": "b.md", "page": 1}
            ],
            "vectors": np.array([
                [1.0, 0.0],
                [0.0, 1.0]
            ], dtype=np.float32)
        }

        results = rag_core.retrieve(
            FakeClient(), "test question", index
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "Relevant")

    def test_retrieve_rejects_empty_question(self):
        with self.assertRaises(ValueError):
            rag_core.retrieve(None, "   ", {})

    def test_retrieve_rejects_mismatched_index(self):
        import numpy as np

        index = {
            "chunks": [
                {"text": "Example", "source": "a.md", "page": 1}
            ],
            "vectors": np.empty((0, 768), dtype=np.float32)
        }

        class FakeClient:
            pass

        with self.assertRaises(ValueError):
            rag_core.retrieve(
                FakeClient(), "test question", index
            )

    def test_empty_index_raises_error(self):
        class FakeClient:
            pass

        documents = [{
            "source": "empty.txt",
            "page": 1,
            "text": ""
        }]

        with self.assertRaises(ValueError):
            rag_core.build_index(FakeClient(), documents)


if __name__ == "__main__":
    unittest.main(verbosity=2)
