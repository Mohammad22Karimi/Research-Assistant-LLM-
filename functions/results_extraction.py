"""
Required function 1 - extract_experimental_results()
====================================================
Extracts the experimental metrics reported by each paper (Accuracy,
Precision, Recall, F1, mAP, IoU, ...).

How it works (simple 3-step recipe, reused by the other functions too):
    1. Fetch the relevant chunks of each paper from the vector database
       (here: results / experiments / abstract sections).
    2. Ask the local LLM to read those chunks and output the metrics as a
       small JSON object.
    3. Parse the JSON and collect everything in a pandas DataFrame.
"""

import json
import re

import pandas as pd

from utils import llm, vector_db

# Sections that usually contain reported numbers.  "full text" is included
# as a fallback for papers where section detection failed.
RESULT_SECTIONS = ["results", "experiments", "abstract", "conclusion", "full text"]

# Keep the context sent to the 8B model reasonably small.
MAX_CONTEXT_CHARS = 8000


def extract_for_paper(collection, paper):
    """
    Purpose : Extract the metrics of ONE paper as a dictionary.
    Inputs  : collection - ChromaDB collection.
              paper (str) - paper name.
    Outputs : metrics (dict) - e.g. {"Accuracy": 91.2, "F1": 89.5}
              (empty dict if nothing could be extracted).
    """
    chunks = vector_db.get_chunks(collection, paper=paper, sections=RESULT_SECTIONS)
    context = "\n".join(chunks)[:MAX_CONTEXT_CHARS]
    if not context:
        return {}

    prompt = (
        "Below is text from the results/experiments sections of the research "
        f"paper '{paper}'.\n"
        "Extract the evaluation metrics that the PROPOSED method of this "
        "paper achieves (its main/best result, not baseline results).\n"
        "Typical metrics: Accuracy, Precision, Recall, F1, mAP, IoU, BLEU, "
        "MOTA, Dice, top-1 accuracy...\n"
        "Reply with ONLY a JSON object mapping metric names to numbers, "
        'for example: {"Accuracy": 91.2, "F1": 89.5}\n'
        "If no metric is found, reply with {}.\n\n"
        f"=== TEXT ===\n{context}"
    )
    reply = llm.ask_llm(prompt, system="You extract numeric results from papers. You output only JSON.")

    # The model may wrap the JSON in extra text - grab the first {...} block.
    match = re.search(r"\{[^{}]*\}", reply)
    if not match:
        return {}
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}

    # Keep only entries whose value is a real number.
    metrics = {}
    for name, value in raw.items():
        if isinstance(value, (int, float)):
            metrics[str(name)] = float(value)
    return metrics


def extract_experimental_results():
    """
    Purpose : Extract metrics for ALL papers in the knowledge base.
    Inputs  : none (reads the vector database).
    Outputs : df (pandas.DataFrame) - rows = papers, columns = metrics,
              missing values = NaN.  Example:

                              Accuracy    F1    mAP
                paper_A           91.2  89.5    NaN
                paper_B           93.0   NaN   41.2
    """
    collection = vector_db.get_collection()
    papers = vector_db.list_papers(collection)
    if not papers:
        print("No papers in the database. Build the knowledge base first.")
        return pd.DataFrame()

    all_metrics = {}
    for paper in papers:
        print(f"[extract] reading results of '{paper}' ...")
        all_metrics[paper] = extract_for_paper(collection, paper)

    # DataFrame.from_dict with orient="index" -> one row per paper.
    df = pd.DataFrame.from_dict(all_metrics, orient="index")
    df.index.name = "paper"
    return df
