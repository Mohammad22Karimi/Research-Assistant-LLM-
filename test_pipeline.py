"""
Integration test for the Utils and RAG layers.

This script tests the project in four levels:
    1. PDF processing, chunking, embeddings, and ChromaDB storage.
    2. Semantic retrieval from the vector database.
    3. RAG retrieval, context construction, grounded-answer logic, and fallback.
    4. Router parsing and, optionally, real routing through Ollama.

Expected project structure:

    project/
    ├── papers/
    │   ├── detr.pdf
    │   ├── efficientdet.pdf
    │   ├── yolov4.pdf
    │   └── yolov7.pdf
    ├── utils/
    │   ├── __init__.py
    │   ├── chunking.py
    │   ├── embeddings.py
    │   ├── llm.py
    │   ├── pdf_processing.py
    │   └── vector_db.py
    ├── rag/
    │   ├── __init__.py
    │   ├── pipeline.py
    │   └── router.py
    └── test_pipeline.py

Run the complete test:

    python test_pipeline.py

Run without real LLM calls (useful when Ollama is not running):

    python test_pipeline.py --skip-llm

Use a different papers directory:

    python test_pipeline.py --papers-dir /path/to/papers

Requirements for full testing:
    pip install pymupdf chromadb sentence-transformers ollama
    ollama pull llama3.1:8b
    ollama serve
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Callable, TypeVar
from unittest.mock import patch

try:
    from utils import chunking, embeddings, pdf_processing, vector_db
    from rag import pipeline, router
except ModuleNotFoundError as exc:
    missing = exc.name or "an unknown module"
    raise SystemExit(
        "Could not import the project modules.\n"
        f"Missing module: {missing}\n\n"
        "Make sure test_pipeline.py is located in the project root, that the "
        "utils/ and rag/ folders contain __init__.py files, and that the required "
        "packages are installed:\n"
        "  pip install pymupdf chromadb sentence-transformers ollama"
    ) from exc


T = TypeVar("T")

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PAPERS_DIR = PROJECT_ROOT / "papers"
TEST_DB_DIR = PROJECT_ROOT / "database_test"
TEST_COLLECTION_NAME = "papers_test"

# This is the assignment description, not one of the research papers.
IGNORED_PDF_NAMES = {"پروژه دوم LLM.pdf"}

RETRIEVAL_QUESTION = (
    "Which object detection dataset is used to evaluate the methods in these papers?"
)
RAG_QUESTION = (
    "Which paper uses a transformer architecture for object detection, "
    "and what are the main components of its method?"
)


class TestRunner:
    """Small test runner that keeps the output readable in a terminal."""

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def run(self, title: str, test_function: Callable[[], T]) -> T | None:
        print(f"\n{'=' * 72}\nTEST: {title}\n{'=' * 72}")
        try:
            result = test_function()
        except Exception as exc:  # keep later independent tests running
            self.failed += 1
            print(f"[FAILED] {type(exc).__name__}: {exc}")
            return None

        self.passed += 1
        print("[PASSED]")
        return result

    def print_summary(self) -> None:
        print(f"\n{'=' * 72}")
        print("TEST SUMMARY")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"{'=' * 72}")


def require(condition: bool, message: str) -> None:
    """Raise a readable assertion error when a smoke-test condition fails."""
    if not condition:
        raise AssertionError(message)


def shorten(text: str, limit: int = 900) -> str:
    """Keep long LLM and chunk outputs readable in the terminal."""
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[:limit] + " ..."


def discover_pdfs(papers_dir: Path) -> list[Path]:
    """Find research-paper PDFs while excluding the assignment description."""
    if not papers_dir.exists():
        raise FileNotFoundError(
            f"Papers directory does not exist: {papers_dir}\n"
            "Create a papers/ folder or pass --papers-dir explicitly."
        )
    if not papers_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {papers_dir}")

    pdf_paths = sorted(
        path
        for path in papers_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".pdf"
        and path.name not in IGNORED_PDF_NAMES
    )

    require(pdf_paths, f"No research-paper PDFs were found in {papers_dir}")
    require(
        len(pdf_paths) >= 4,
        f"The assignment requires at least four papers, but only {len(pdf_paths)} were found.",
    )
    return pdf_paths


def configure_isolated_test_database() -> object:
    """
    Point vector_db at a separate test database.

    This prevents the test from clearing or modifying the real project database.
    pipeline.retrieve() uses the same imported vector_db module, so the RAG layer
    automatically reads from this isolated test collection too.
    """
    if TEST_DB_DIR.exists():
        shutil.rmtree(TEST_DB_DIR)

    vector_db.DB_PATH = str(TEST_DB_DIR)
    vector_db.COLLECTION_NAME = TEST_COLLECTION_NAME

    collection = vector_db.get_collection()
    vector_db.clear(collection)
    return collection


def build_knowledge_base(collection: object, papers_dir: Path) -> list[dict]:
    """Run Stages 1-4 on every paper and populate the isolated test database."""
    pdf_paths = discover_pdfs(papers_dir)
    print(f"Papers directory: {papers_dir}")
    print(f"Research papers found: {len(pdf_paths)}")

    all_records: list[dict] = []

    for pdf_path in pdf_paths:
        paper_name, sections = pdf_processing.process_pdf(str(pdf_path))
        require(sections, f"No text sections were extracted from {pdf_path.name}")

        chunks = chunking.chunk_sections(paper_name, sections)
        require(chunks, f"No chunks were generated for {pdf_path.name}")
        all_records.extend(chunks)

        print(
            f"  - {paper_name}: {len(sections)} sections, "
            f"{len(chunks)} chunks"
        )

    require(all_records, "The complete chunk list is empty.")

    print(f"Generating embeddings for {len(all_records)} chunks...")
    vectors = embeddings.embed_texts([record["text"] for record in all_records])

    require(
        len(vectors) == len(all_records),
        "The number of embeddings does not match the number of chunks.",
    )
    require(vectors and len(vectors[0]) > 0, "The embedding vectors are empty.")

    vector_db.add_chunks(collection, all_records, vectors)

    stored_count = collection.count()
    require(
        stored_count == len(all_records),
        f"Expected {len(all_records)} stored chunks, but ChromaDB contains {stored_count}.",
    )

    stored_papers = vector_db.list_papers(collection)
    require(
        len(stored_papers) == len(pdf_paths),
        "The number of paper names in ChromaDB does not match the input PDFs.",
    )

    print(f"Stored chunks: {stored_count}")
    print(f"Stored papers: {', '.join(stored_papers)}")
    return all_records


def print_hits(hits: list[dict], preview_length: int = 260) -> None:
    """Print retrieval results with source metadata."""
    for index, hit in enumerate(hits, start=1):
        print(
            f"\nResult {index} | similarity={hit['similarity']:.3f} | "
            f"paper={hit['paper']} | section={hit['section']}"
        )
        print(shorten(hit["text"], preview_length))


def test_utils_semantic_search(collection: object) -> list[dict]:
    """Test embedding a question and searching ChromaDB directly."""
    print(f"Question: {RETRIEVAL_QUESTION}")
    question_vector = embeddings.embed_text(RETRIEVAL_QUESTION)
    hits = vector_db.search(collection, question_vector, top_k=4)

    require(hits, "Vector search returned no results.")
    require(len(hits) <= 4, "Vector search returned more than top_k results.")

    required_keys = {"text", "paper", "section", "similarity"}
    for hit in hits:
        require(
            required_keys.issubset(hit),
            f"A retrieval result is missing fields: {required_keys - set(hit)}",
        )

    print_hits(hits)
    return hits


def test_rag_retrieval_and_context() -> list[dict]:
    """Test rag.pipeline.retrieve() and rag.pipeline.build_context()."""
    print(f"Question: {RAG_QUESTION}")
    hits = pipeline.retrieve(RAG_QUESTION, top_k=4)
    require(hits, "RAG retrieval returned no chunks.")

    context = pipeline.build_context(hits)
    require(context.strip(), "RAG context is empty.")
    require("[Chunk 1 | paper:" in context, "Chunk source labels are missing.")
    require("| section:" in context, "Section metadata is missing from context.")

    for hit in hits:
        require(hit["paper"] in context, "A retrieved paper name is absent from context.")
        require(hit["section"] in context, "A retrieved section is absent from context.")

    print_hits(hits)
    print("\nContext preview:")
    print(shorten(context, 1200))
    return hits


def test_rag_branch_logic_without_llm() -> None:
    """
    Deterministically test both ask() branches without embeddings, ChromaDB,
    or Ollama. This verifies the threshold decision and response formatting.
    """
    high_similarity_hit = {
        "text": "DETR uses a transformer encoder-decoder architecture.",
        "paper": "detr",
        "section": "methodology",
        "similarity": pipeline.SIMILARITY_THRESHOLD + 0.10,
    }
    low_similarity_hit = {
        "text": "Unrelated text.",
        "paper": "detr",
        "section": "introduction",
        "similarity": max(0.0, pipeline.SIMILARITY_THRESHOLD - 0.10),
    }

    with (
        patch.object(pipeline, "retrieve", return_value=[high_similarity_hit]),
        patch.object(
            pipeline,
            "answer_from_papers",
            return_value="Mock grounded answer from the retrieved paper.",
        ) as grounded_mock,
        patch.object(pipeline, "answer_general") as general_mock,
    ):
        response = pipeline.ask("A question covered by the papers")
        require("=== Answer from papers ===" in response, "Grounded-answer marker is missing.")
        require("Mock grounded answer" in response, "Grounded answer is missing.")
        grounded_mock.assert_called_once()
        general_mock.assert_not_called()

    with (
        patch.object(pipeline, "retrieve", return_value=[low_similarity_hit]),
        patch.object(pipeline, "answer_from_papers") as grounded_mock,
        patch.object(
            pipeline,
            "answer_general",
            return_value="Mock general-knowledge answer.",
        ) as general_mock,
    ):
        response = pipeline.ask("A question not covered by the papers")
        require(
            "The answer was not found in the selected papers." in response,
            "Fallback notice is missing.",
        )
        require(
            "=== General LLM answer (NOT from the papers) ===" in response,
            "General-answer marker is missing.",
        )
        require("Mock general-knowledge answer" in response, "Fallback answer is missing.")
        grounded_mock.assert_not_called()
        general_mock.assert_called_once()

    print("Both the grounded-answer and fallback branches behaved correctly.")


def test_rag_end_to_end_with_ollama() -> str:
    """Run the complete Question -> Retrieval -> Context -> LLM path."""
    hits = pipeline.retrieve(RAG_QUESTION, top_k=pipeline.TOP_K)
    require(hits, "No chunks were retrieved for the end-to-end RAG question.")

    best_similarity = hits[0]["similarity"]
    print(f"Best similarity: {best_similarity:.3f}")
    print(f"Configured threshold: {pipeline.SIMILARITY_THRESHOLD:.3f}")

    require(
        best_similarity >= pipeline.SIMILARITY_THRESHOLD,
        "The clearly in-scope RAG question fell below the similarity threshold. "
        "Inspect retrieval quality or recalibrate SIMILARITY_THRESHOLD.",
    )

    response = pipeline.ask(RAG_QUESTION)
    require(response.strip(), "The RAG pipeline returned an empty answer.")
    require(
        "=== Answer from papers ===" in response,
        "The in-scope question unexpectedly used the fallback branch.",
    )

    print("\nRAG answer:")
    print(shorten(response, 2200))
    return response


def test_router_parsing_without_llm() -> None:
    """Test router keyword extraction and safe default deterministically."""
    with patch.object(router.llm, "ask_llm", return_value="COMPARE_RESULTS."):
        require(
            router.route("Compare all papers") == "COMPARE_RESULTS",
            "Router did not extract COMPARE_RESULTS from a punctuated reply.",
        )

    with patch.object(router.llm, "ask_llm", return_value="I am not sure"):
        require(
            router.route("Ambiguous request") == "ANSWER",
            "Router did not fall back to ANSWER for an invalid model reply.",
        )

    print("Router parsing and safe-default logic behaved correctly.")


def test_router_with_ollama() -> None:
    """Check whether the real local LLM selects the expected action."""
    cases = [
        ("Which dataset is used by DETR?", "ANSWER"),
        ("Compare the experimental results of all four papers.", "COMPARE_RESULTS"),
        ("Write a mini survey of the selected papers.", "GENERATE_SURVEY"),
        ("Find research gaps and future work in these papers.", "RESEARCH_GAP"),
    ]

    mismatches: list[str] = []
    for question, expected in cases:
        actual = router.route(question)
        print(f"Expected={expected:<18} Actual={actual:<18} Question={question}")
        if actual != expected:
            mismatches.append(f"{question!r}: expected {expected}, got {actual}")

    require(
        not mismatches,
        "The real LLM router produced unexpected actions:\n  - " + "\n  - ".join(mismatches),
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Utils + RAG + Router.")
    parser.add_argument(
        "--papers-dir",
        type=Path,
        default=DEFAULT_PAPERS_DIR,
        help="Directory containing the research-paper PDFs.",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip tests that make real calls to Ollama.",
    )
    parser.add_argument(
        "--skip-router",
        action="store_true",
        help="Skip router tests.",
    )
    return parser.parse_args()


def prepare_knowledge_base(papers_dir: Path) -> object:
    """Create the isolated collection, fill it, and return it for later tests."""
    collection = configure_isolated_test_database()
    build_knowledge_base(collection, papers_dir)
    return collection


def main() -> int:
    args = parse_arguments()
    papers_dir = args.papers_dir.expanduser().resolve()
    runner = TestRunner()

    collection = runner.run(
        "Build isolated knowledge base with Utils (Stages 1-4)",
        lambda: prepare_knowledge_base(papers_dir),
    )

    # Building the database is a prerequisite for retrieval and real RAG tests.
    if collection is not None:
        runner.run(
            "Direct semantic search through Utils",
            lambda: test_utils_semantic_search(collection),
        )
        runner.run("RAG retrieval and context construction", test_rag_retrieval_and_context)

    runner.run("RAG threshold branches without Ollama", test_rag_branch_logic_without_llm)

    if not args.skip_llm and collection is not None:
        runner.run("End-to-end RAG answer through Ollama", test_rag_end_to_end_with_ollama)
    elif args.skip_llm:
        print("\n[SKIPPED] End-to-end RAG answer through Ollama (--skip-llm)")

    if not args.skip_router:
        runner.run("Router parsing without Ollama", test_router_parsing_without_llm)
        if not args.skip_llm:
            runner.run("Router decisions through Ollama", test_router_with_ollama)
        else:
            print("\n[SKIPPED] Router decisions through Ollama (--skip-llm)")
    else:
        print("\n[SKIPPED] Router tests (--skip-router)")

    runner.print_summary()

    if runner.failed:
        print(
            "\nOne or more tests failed. For LLM-related failures, confirm that "
            "Ollama is running and llama3.1:8b has been pulled."
        )
        return 1

    print("\nAll executed tests passed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())