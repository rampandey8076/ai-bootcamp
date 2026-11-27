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


def test_retry_succeeds_after_temporary_failure(monkeypatch):
    calls = {"count": 0}

    def request():
        calls["count"] += 1
        if calls["count"] < 3:
            raise Exception("503 Service unavailable")
        return "success"

    monkeypatch.setattr(rag_core.time, "sleep", lambda _: None)

    result = rag_core.call_with_retry(
        "test operation", request
    )

    assert result == "success"
    assert calls["count"] == 3


def test_retry_raises_after_maximum_attempts(monkeypatch):
    calls = {"count": 0}
    delays = []

    def request():
        calls["count"] += 1
        raise Exception("503 Service unavailable")

    monkeypatch.setattr(
        rag_core.time, "sleep", lambda delay: delays.append(delay)
    )

    import pytest

    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        rag_core.call_with_retry("test operation", request)

    assert calls["count"] == 3
    assert delays == [1, 2]


def test_retry_does_not_retry_permanent_errors(monkeypatch):
    calls = {"count": 0}

    def request():
        calls["count"] += 1
        raise ValueError("Invalid request")

    monkeypatch.setattr(rag_core.time, "sleep", lambda _: None)

    import pytest

    with pytest.raises(ValueError, match="Invalid request"):
        rag_core.call_with_retry("test operation", request)

    assert calls["count"] == 1


def test_unanswerable_question_returns_fallback(monkeypatch):
    monkeypatch.setattr(
        rag_core, "retrieve", lambda *args: []
    )

    answer, sources = rag_core.ask_question(
        None, "Unrelated question", {}
    )

    assert answer == rag_core.FALLBACK
    assert sources == []


def test_answer_generation_preserves_source_pages(monkeypatch):
    from types import SimpleNamespace

    results = [
        {
            "text": "Embeddings represent text meaning.",
            "source": "embeddings.md",
            "page": 1,
            "score": 0.91
        },
        {
            "text": "Vector databases store vectors.",
            "source": "vectors.md",
            "page": 2,
            "score": 0.85
        }
    ]

    monkeypatch.setattr(
        rag_core, "retrieve", lambda *args: results
    )

    class FakeModels:
        def generate_content(self, **kwargs):
            assert "[1] File: embeddings.md, Page: 1" in kwargs["contents"]
            assert "[2] File: vectors.md, Page: 2" in kwargs["contents"]

            return SimpleNamespace(
                text="Embeddings represent meaning [1]."
            )

    class FakeClient:
        models = FakeModels()

    answer, sources = rag_core.ask_question(
        FakeClient(), "What are embeddings?", {}
    )

    assert "[1]" in answer
    assert len(sources) == 2
    assert sources[0]["source"] == "embeddings.md"
    assert sources[1]["source"] == "vectors.md"


def test_retry_validates_max_attempts():
    import pytest

    with pytest.raises(ValueError):
        rag_core.call_with_retry("test", lambda: "success", 0)

    with pytest.raises(ValueError):
        rag_core.call_with_retry("test", lambda: "success", -1)


def test_retry_does_not_match_unrelated_numbers(monkeypatch):
    import pytest

    calls = {"count": 0}

    def request():
        calls["count"] += 1
        raise Exception("Document contains the number 1500")

    monkeypatch.setattr(rag_core.time, "sleep", lambda _: None)

    with pytest.raises(Exception, match="1500"):
        rag_core.call_with_retry("test", request)

    assert calls["count"] == 1


def test_answer_generation_removes_duplicate_source_pages(monkeypatch):
    from types import SimpleNamespace

    results = [
        {
            "text": "First chunk",
            "source": "guide.md",
            "page": 1,
            "score": 0.95
        },
        {
            "text": "Second chunk",
            "source": "guide.md",
            "page": 1,
            "score": 0.90
        },
        {
            "text": "Third chunk",
            "source": "guide.md",
            "page": 2,
            "score": 0.85
        }
    ]

    monkeypatch.setattr(
        rag_core, "retrieve", lambda *args: results
    )

    class FakeModels:
        def generate_content(self, **kwargs):
            return SimpleNamespace(text="Answer [1].")

    class FakeClient:
        models = FakeModels()

    answer, sources = rag_core.ask_question(
        FakeClient(), "Explain the guide.", {}
    )

    assert answer == "Answer [1]."
    assert len(sources) == 2
    assert [(s["source"], s["page"]) for s in sources] == [
        ("guide.md", 1),
        ("guide.md", 2)
    ]


def test_citations_map_to_unique_source_pages(monkeypatch):
    from types import SimpleNamespace

    results = [
        {
            "text": "First part of the guide.",
            "source": "guide.pdf",
            "page": 1,
            "score": 0.95
        },
        {
            "text": "Second part of the guide.",
            "source": "guide.pdf",
            "page": 1,
            "score": 0.90
        },
        {
            "text": "Information from another page.",
            "source": "guide.pdf",
            "page": 2,
            "score": 0.85
        }
    ]

    monkeypatch.setattr(
        rag_core, "retrieve", lambda *args: results
    )

    captured = {}

    class FakeModels:
        def generate_content(self, **kwargs):
            captured["prompt"] = kwargs["contents"]
            return SimpleNamespace(
                text="The guide explains the topic [1]."
            )

    class FakeClient:
        models = FakeModels()

    answer, sources = rag_core.ask_question(
        FakeClient(), "Explain the guide.", {}
    )

    prompt = captured["prompt"]

    assert "[1] File: guide.pdf, Page: 1" in prompt
    assert "[2] File: guide.pdf, Page: 2" in prompt

    assert "First part of the guide." in prompt
    assert "Second part of the guide." in prompt

    assert prompt.count("[1] File:") == 1
    assert prompt.count("[2] File:") == 1
    assert "[3] File:" not in prompt

    assert answer == "The guide explains the topic [1]."

    assert len(sources) == 2
    assert [
        (source["source"], source["page"])
        for source in sources
    ] == [
        ("guide.pdf", 1),
        ("guide.pdf", 2)
    ]

    assert sources[0]["score"] == 0.95
    assert sources[1]["score"] == 0.85
