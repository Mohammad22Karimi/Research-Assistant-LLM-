"""
Required function 3 - generate_survey()
=======================================
Generates a "mini survey" over all papers in the knowledge base with the
five sections required by the project:

    1. Problem introduction
    2. Summary of each paper
    3. Comparison of the methods
    4. Evolution of the methods over time
    5. Conclusion

Strategy (kept deliberately simple):
    * First summarize every paper individually from its abstract /
      introduction / conclusion chunks (small prompts work much better
      with an 8B model than one giant prompt with 4 full papers).
    * Then give ALL the summaries to the LLM once more and ask it to write
      the full survey document.
"""

from utils import llm, vector_db

SUMMARY_SECTIONS = ["abstract", "introduction", "methodology", "conclusion", "full text"]
MAX_CONTEXT_CHARS = 8000
OUTPUT_FILE = "mini_survey.md"


def summarize_paper(collection, paper):
    """
    Purpose : Produce a short structured summary of ONE paper.
    Inputs  : collection - ChromaDB collection.
              paper (str) - paper name.
    Outputs : summary (str) - a paragraph covering problem, method, results.
    """
    chunks = vector_db.get_chunks(collection, paper=paper, sections=SUMMARY_SECTIONS)
    context = "\n".join(chunks)[:MAX_CONTEXT_CHARS]

    prompt = (
        f"Summarize the research paper '{paper}' in about 150 words.\n"
        "Cover: (a) the problem it addresses, (b) the proposed method, "
        "(c) the main results.\n\n"
        f"=== PAPER TEXT ===\n{context}"
    )
    return llm.ask_llm(prompt, system="You write concise academic summaries.")


def generate_survey():
    """
    Purpose : Generate the mini survey over all stored papers.
    Inputs  : none (reads the vector database).
    Outputs : survey (str) - the survey text.
              Side effect: the survey is also saved to "mini_survey.md".
    """
    collection = vector_db.get_collection()
    papers = vector_db.list_papers(collection)
    if not papers:
        print("No papers in the database. Build the knowledge base first.")
        return ""

    # ---- step 1: one summary per paper ----
    summaries = []
    for paper in papers:
        print(f"[survey] summarizing '{paper}' ...")
        summaries.append(f"### {paper}\n{summarize_paper(collection, paper)}")
    combined = "\n\n".join(summaries)

    # ---- step 2: write the survey from the summaries ----
    print("[survey] writing the final survey ...")
    prompt = (
        "Using the paper summaries below, write a mini survey in Markdown "
        "with EXACTLY these five sections:\n"
        "1. Introduction to the problem\n"
        "2. Summary of each paper\n"
        "3. Comparison of the methods\n"
        "4. Evolution of the methods\n"
        "5. Conclusion\n\n"
        f"=== PAPER SUMMARIES ===\n{combined}"
    )
    survey = llm.ask_llm(prompt, system="You are an academic survey writer.")

    # Save to disk so the student can show/submit the generated document.
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(survey)
    print(f"Survey saved to: {OUTPUT_FILE}")

    return survey
