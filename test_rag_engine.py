import unittest
from unittest.mock import patch

from rag_engine import (
    CHUNK_SIZE,
    FinancialLiteracyRAG,
    _generate_answer,
    _get_seeded_records,
    _remove_page_references,
    _split_text,
    generation_is_configured,
)


class FakeCollection:
    def __init__(
        self,
        documents: list[str],
        metadatas: list[dict[str, str]],
        distances: list[float] | None = None,
    ):
        self.documents = documents
        self.metadatas = metadatas
        self.distances = distances or [0.2] * len(documents)
        self.queries: list[dict[str, object]] = []

    def count(self) -> int:
        return len(self.documents)

    def query(self, **kwargs: object) -> dict[str, list[list[object]]]:
        self.queries.append(kwargs)
        where = kwargs.get("where")
        requested_language = (
            where.get("language") if isinstance(where, dict) else None
        )
        matches = [
            (document, metadata, distance)
            for document, metadata, distance in zip(
                self.documents, self.metadatas, self.distances
            )
            if requested_language is None
            or metadata.get("language", "English") == requested_language
        ]
        return {
            "documents": [[document for document, _, _ in matches]],
            "metadatas": [[metadata for _, metadata, _ in matches]],
            "distances": [[distance for _, _, distance in matches]],
        }


class FinancialLiteracyRAGTests(unittest.TestCase):
    def make_rag(
        self,
        documents: list[str],
        metadatas: list[dict[str, str]],
        distances: list[float] | None = None,
    ) -> FinancialLiteracyRAG:
        rag = object.__new__(FinancialLiteracyRAG)
        rag.collection = FakeCollection(documents, metadatas, distances)
        return rag

    @patch("rag_engine.embed_texts", return_value=[[0.1, 0.2]])
    def test_answer_with_sources_filters_by_language(self, _embed: object) -> None:
        rag = self.make_rag(
            ["English guidance", "हिंदी जानकारी", "अधिक हिंदी जानकारी"],
            [
                {"language": "English", "source": "English.txt"},
                {"language": "Hindi", "source": "Hindi.txt"},
                {"language": "Hindi", "source": "Hindi.txt"},
            ],
        )

        answer, sources = rag.answer_with_sources("यूपीआई", "Hindi", top_k=2)

        self.assertIn("हिंदी जानकारी", answer)
        self.assertNotIn("English guidance", answer)
        self.assertEqual(sources, ["Hindi.txt"])

    @patch("rag_engine.embed_texts", return_value=[[0.1, 0.2]])
    def test_answer_falls_back_to_english_with_source(self, _embed: object) -> None:
        rag = self.make_rag(
            ["English guidance"],
            [{"language": "English", "source": "Guide.pdf"}],
        )

        answer, sources = rag.answer_with_sources("question", "Marathi")

        self.assertEqual(answer, "• English guidance")
        self.assertEqual(sources, ["Guide.pdf"])

    @patch("rag_engine.embed_texts", return_value=[[0.1, 0.2]])
    def test_answer_drops_distant_chunks(self, _embed: object) -> None:
        rag = self.make_rag(
            ["Relevant answer", "Unrelated document"],
            [
                {"language": "English", "source": "Guide.txt"},
                {"language": "English", "source": "Unrelated.pdf"},
            ],
            [0.2, 0.5],
        )

        answer, sources = rag.answer_with_sources("question")

        self.assertEqual(answer, "• Relevant answer")
        self.assertEqual(sources, ["Guide.txt"])

    @patch("rag_engine.embed_texts", return_value=[[0.1, 0.2]])
    def test_answer_falls_back_when_language_matches_are_irrelevant(
        self, _embed: object
    ) -> None:
        rag = self.make_rag(
            ["Relevant English guidance", "Unrelated Hindi guidance"],
            [
                {"language": "English", "source": "English.txt"},
                {"language": "Hindi", "source": "Hindi.txt"},
            ],
            [0.2, 0.8],
        )

        answer, sources = rag.answer_with_sources("question", "Hindi")

        self.assertEqual(answer, "• Relevant English guidance")
        self.assertEqual(sources, ["English.txt"])
        self.assertEqual(
            [query["where"] for query in rag.collection.queries],
            [{"language": "Hindi"}, {"language": "English"}],
        )

    def test_answers_remove_page_references(self) -> None:
        text = (
            "Page 12 of 20\nCompare the total cost (p. 7) before signing.\n"
            "54 | P a g e amount claimed as deduction.\n"
            "23 CHAPTER 6 - INSURANCE RELATED PRODUCTS..................... 38\n"
            "CHAPTER 7 - PENSION PLANNING................................. 40"
        )

        self.assertEqual(
            _remove_page_references(text),
            "Compare the total cost before signing.\namount claimed as deduction.",
        )

    def test_watsonx_answer_is_conversational_and_page_free(self) -> None:
        class FakeModel:
            def generate_text(self, *, prompt: str) -> str:
                self.prompt = prompt
                return "Compare the APR and total repayment (page 4)."

        model = FakeModel()
        with patch("rag_engine._get_generation_model", return_value=model):
            answer = _generate_answer(
                "How should I compare loans?",
                "Compare APR and total repayment.",
                "English",
            )

        self.assertEqual(answer, "Compare the APR and total repayment.")
        self.assertIn("How should I compare loans?", model.prompt)
        self.assertIn("Do not mention page numbers", model.prompt)

    def test_watsonx_requires_api_key_and_project_id(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(generation_is_configured())

        with patch.dict("os.environ", {"WATSONX_APIKEY": "key"}, clear=True):
            self.assertFalse(generation_is_configured())

        with patch.dict(
            "os.environ",
            {"WATSONX_APIKEY": "key", "WATSONX_PROJECT_ID": "project"},
            clear=True,
        ):
            self.assertTrue(generation_is_configured())

    def test_seeded_guidance_covers_all_supported_languages(self) -> None:
        records = _get_seeded_records()
        languages = {record["language"] for record in records}
        self.assertEqual(languages, {"English", "Hindi", "Tamil", "Telugu", "Bengali", "Marathi"})

        for language in languages:
            sources = {record["source"] for record in records if record["language"] == language}
            code = self.language_code(language)
            for topic in (
                "UPI_Transfer",
                "Loan_Interest",
                "QR_Reward_Scam",
                "Online_Scams",
                "Digital_Payments",
                "Budgeting",
                "Govt_Schemes",
            ):
                self.assertIn(f"{topic}_{code}.txt", sources)
        qr_guidance = [
            record for record in records
            if record["source"].startswith("QR_Reward_Scam_")
        ]
        self.assertTrue(all(len(record["text"]) <= 600 for record in qr_guidance))

    def test_chunks_respect_size_with_and_without_overlap(self) -> None:
        text = "financial literacy " * 100

        for overlap in (0, 80):
            with self.subTest(overlap=overlap):
                chunks = _split_text(text, chunk_size=120, overlap=overlap)
                self.assertTrue(chunks)
                self.assertTrue(all(len(chunk) <= 120 for chunk in chunks))

    def test_seeded_guidance_chunks_respect_configured_limit(self) -> None:
        for record in _get_seeded_records():
            with self.subTest(source=record["source"]):
                chunks = _split_text(record["text"])
                self.assertTrue(chunks)
                self.assertTrue(all(len(chunk) <= CHUNK_SIZE for chunk in chunks))

    def test_chunker_rejects_invalid_sizes(self) -> None:
        with self.assertRaises(ValueError):
            _split_text("text", chunk_size=0)
        with self.assertRaises(ValueError):
            _split_text("text", chunk_size=10, overlap=10)

    @staticmethod
    def language_code(language: str) -> str:
        return {
            "English": "en",
            "Hindi": "hi",
            "Tamil": "ta",
            "Telugu": "te",
            "Bengali": "bn",
            "Marathi": "mr",
        }[language]


if __name__ == "__main__":
    unittest.main()
