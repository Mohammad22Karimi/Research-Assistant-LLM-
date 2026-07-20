"""
Bonus function 1 - find_research_gap()
======================================
Analyzes the conclusion / future-work parts of every paper and extracts:
    * limitations
    * unsolved problems
    * suggestions for future research

Same 3-step recipe as the other functions:
fetch chunks -> ask the LLM -> print/return the result.
"""

from utils import llm, vector_db

GAP_SECTIONS = ["conclusion", "results", "full text"]
MAX_CONTEXT_CHARS = 6000


def find_research_gap():
    """
    Purpose : Extract limitations / unsolved problems / future work for
              every paper and summarize the overall research gap.
    Inputs  : none (reads the vector database).
    Outputs : report (str) - the research gap analysis text.
    """
    collection = vector_db.get_collection()
    papers = vector_db.list_papers(collection)
    if not papers:
        print("No papers in the database. Build the knowledge base first.")
        return ""

    per_paper = []
    for paper in papers:
        print(f"[gap] analyzing '{paper}' ...")
        chunks = vector_db.get_chunks(collection, paper=paper, sections=GAP_SECTIONS)
        context = "\n".join(chunks)[:MAX_CONTEXT_CHARS]

        prompt = (
            f"The text below is from the conclusion/results of the paper "
            f"'{paper}'.\nExtract, as short bullet points:\n"
            "- Limitations of the method\n"
            "- Unsolved problems\n"
            "- Suggestions for future research\n\n"
            f"=== TEXT ===\n{context}"
        )
        analysis = llm.ask_llm(prompt, system="You analyze research papers for limitations and open problems.")
        per_paper.append(f"### {paper}\n{analysis}")

    combined = "\n\n".join(per_paper)

    # One final call that looks across ALL papers to state the common gap.
    prompt = (
        "Based on the per-paper analyses below, write a short paragraph "
        "describing the overall RESEARCH GAP in this area - what none of "
        "the papers has solved yet.\n\n" + combined
    )
    overall = llm.ask_llm(prompt, system="You identify research gaps.")

    report = combined + "\n\n### Overall research gap\n" + overall
    print("\n" + report)
    return report
