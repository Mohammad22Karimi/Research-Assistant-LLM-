"""
Bonus function 3 - compare_limitations()
========================================
Extracts the limitations of each paper's method and then compares the
weaknesses of the different approaches side by side.
"""

from utils import llm, vector_db

LIMITATION_SECTIONS = ["conclusion", "results", "experiments", "full text"]
MAX_CONTEXT_CHARS = 6000


def compare_limitations():
    """
    Purpose : Extract each paper's limitations and produce a comparison of
              the weak points of the different methods.
    Inputs  : none (reads the vector database).
    Outputs : report (str) - per-paper limitations + comparison text.
    """
    collection = vector_db.get_collection()
    papers = vector_db.list_papers(collection)
    if not papers:
        print("No papers in the database. Build the knowledge base first.")
        return ""

    # ---- step 1: limitations of each paper separately ----
    per_paper = []
    for paper in papers:
        print(f"[limitations] analyzing '{paper}' ...")
        chunks = vector_db.get_chunks(collection, paper=paper, sections=LIMITATION_SECTIONS)
        context = "\n".join(chunks)[:MAX_CONTEXT_CHARS]

        prompt = (
            f"List, as short bullet points, the LIMITATIONS / weaknesses of "
            f"the method proposed in the paper '{paper}', based on the text "
            f"below.\n\n=== TEXT ===\n{context}"
        )
        analysis = llm.ask_llm(prompt, system="You analyze weaknesses of research methods.")
        per_paper.append(f"### {paper}\n{analysis}")

    combined = "\n\n".join(per_paper)

    # ---- step 2: compare the weaknesses across papers ----
    prompt = (
        "Below are the limitations of several methods that solve the same "
        "problem. Write a short comparison: which weaknesses are shared by "
        "all methods, and which are specific to one method?\n\n" + combined
    )
    comparison = llm.ask_llm(prompt, system="You compare weaknesses of research methods.")

    report = combined + "\n\n### Comparison of weaknesses\n" + comparison
    print("\n" + report)
    return report
