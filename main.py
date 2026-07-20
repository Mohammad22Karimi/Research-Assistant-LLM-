"""
Stage 8 - CLI (terminal user interface)
=======================================
Entry point of the Research Assistant.  Shows a numbered menu, reads the
user's choice with input(), and calls the matching part of the system.

Run with:
    python main.py

"Ask Question" (option 2) demonstrates Stage 6 function calling:
the question first goes through the LLM router, which decides whether to
answer with plain RAG or to call one of the analysis functions.
"""

import glob
import os

from functions.comparison import compare_results
from functions.limitations import compare_limitations
from functions.references import compare_references
from functions.research_gap import find_research_gap
from functions.results_extraction import extract_experimental_results
from functions.survey import generate_survey
from rag import pipeline, router
from utils import chunking, embeddings, pdf_processing, vector_db

PAPERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "papers")


def build_knowledge_base():
    """
    Purpose : Run Stages 1-4 for every PDF in the papers/ folder:
              extract text -> detect sections -> chunk -> embed -> store.
    Inputs  : none (reads PDFs from papers/).
    Outputs : none (fills the ChromaDB database on disk).
    """
    pdf_files = sorted(glob.glob(os.path.join(PAPERS_DIR, "*.pdf")))
    if not pdf_files:
        print(f"No PDF files found in: {PAPERS_DIR}")
        print("Copy your papers there and try again.")
        return

    collection = vector_db.get_collection()
    vector_db.clear(collection)  # rebuild from scratch to avoid stale data

    for pdf_path in pdf_files:
        # Stage 1: PDF -> text -> sections
        paper_name, sections = pdf_processing.process_pdf(pdf_path)
        print(f"\n[ingest] '{paper_name}': found sections -> {list(sections.keys())}")

        # Stage 2: sections -> overlapping chunks with metadata
        records = chunking.chunk_sections(paper_name, sections)
        print(f"[ingest] created {len(records)} chunks")

        # Stage 3: chunks -> embeddings
        vectors = embeddings.embed_texts([r["text"] for r in records])

        # Stage 4: store chunks + metadata + embeddings in ChromaDB
        vector_db.add_chunks(collection, records, vectors)

    print(f"\nKnowledge base ready: {len(vector_db.list_papers(collection))} papers stored.")


def ask_question():
    """
    Purpose : Handle one free-form user question (Stages 5, 6 and 7).
              The LLM router decides whether the question is a normal RAG
              question or a request that should trigger one of the
              analysis functions.
    Inputs  : none (reads the question with input()).
    Outputs : none (prints the answer).
    """
    question = input("\nAsk your question:\n> ").strip()
    if not question:
        return

    # Stage 6: the LLM chooses the action (function calling).
    action = router.route(question)
    print(f"[router] LLM chose action: {action}")

    # Plain-Python dispatch - each keyword maps to one function.
    if action == "EXTRACT_RESULTS":
        df = extract_experimental_results()
        if not df.empty:
            print("\n" + df.to_string())
    elif action == "COMPARE_RESULTS":
        compare_results()
    elif action == "GENERATE_SURVEY":
        print("\n" + generate_survey())
    elif action == "RESEARCH_GAP":
        find_research_gap()
    elif action == "COMPARE_REFERENCES":
        compare_references()
    elif action == "COMPARE_LIMITATIONS":
        compare_limitations()
    else:
        # Stage 5 RAG answer, with the Stage 7 fallback built in.
        print("\n" + pipeline.ask(question))


MENU = """
==============================
     Research Assistant
==============================
1. Build / update knowledge base
2. Ask question  (LLM decides: RAG answer or function call)
3. Extract experimental results
4. Compare results (table + chart)
5. Generate mini survey
6. Find research gap        [bonus]
7. Compare references       [bonus]
8. Compare limitations      [bonus]
9. Exit
"""


def main():
    """
    Purpose : Main menu loop of the CLI application.
    Inputs  : none (interactive).
    Outputs : none.
    """
    while True:
        print(MENU)
        choice = input("Select an option (1-9): ").strip()

        if choice == "1":
            build_knowledge_base()
        elif choice == "2":
            ask_question()
        elif choice == "3":
            df = extract_experimental_results()
            if not df.empty:
                print("\n" + df.to_string())
        elif choice == "4":
            compare_results()
        elif choice == "5":
            print("\n" + generate_survey())
        elif choice == "6":
            find_research_gap()
        elif choice == "7":
            compare_references()
        elif choice == "8":
            compare_limitations()
        elif choice == "9":
            print("Goodbye!")
            break
        else:
            print("Invalid option, please enter a number from 1 to 9.")


if __name__ == "__main__":
    main()
