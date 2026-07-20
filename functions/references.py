"""
Bonus function 2 - compare_references()
=======================================
Analyzes the References section of every paper and builds a table that
counts how many cited works were published in each year, e.g.:

    Year   Paper A   Paper B   Paper C   Paper D
    2020        12        10         8        15
    2021         9        14        11         7

No LLM is needed here: publication years in reference lists are 4-digit
numbers, so a regular expression + counting is enough (and 100% reliable).
"""

import re
from collections import Counter

import matplotlib

matplotlib.use("Agg")  # save charts to file - works even without a display
import matplotlib.pyplot as plt
import pandas as pd

from utils import vector_db

# Years like 1990-2029.  Word boundaries avoid matching page numbers such
# as "12016".
YEAR_PATTERN = re.compile(r"\b(19[0-9]{2}|20[0-2][0-9])\b")

CHART_FILE = "references_chart.png"


def count_reference_years(references_text):
    """
    Purpose : Count publication years inside one references section.
    Inputs  : references_text (str).
    Outputs : counts (Counter) - year (int) -> number of occurrences.
    """
    years = [int(y) for y in YEAR_PATTERN.findall(references_text)]
    return Counter(years)


def compare_references():
    """
    Purpose : Build and print the year-by-year reference comparison table
              for all papers, plus a line chart.
    Inputs  : none (reads the vector database).
    Outputs : df (pandas.DataFrame) - rows = years, columns = papers,
              values = number of references from that year.
    """
    collection = vector_db.get_collection()
    papers = vector_db.list_papers(collection)
    if not papers:
        print("No papers in the database. Build the knowledge base first.")
        return pd.DataFrame()

    table = {}
    for paper in papers:
        chunks = vector_db.get_chunks(collection, paper=paper, sections=["references"])
        if not chunks:
            print(f"[references] warning: no 'references' section found for '{paper}'")
            continue
        table[paper] = count_reference_years("\n".join(chunks))

    if not table:
        print("No references sections found in any paper.")
        return pd.DataFrame()

    # DataFrame: rows = years (sorted), columns = papers, missing -> 0
    df = pd.DataFrame(table).fillna(0).astype(int).sort_index()
    df.index.name = "Year"

    print("\n=== References per publication year ===")
    print(df.to_string())

    # A simple line chart shows how "recent" each paper's bibliography is.
    ax = df.plot(kind="line", marker="o", figsize=(10, 6))
    ax.set_title("Number of cited works per publication year")
    ax.set_xlabel("Publication year of cited work")
    ax.set_ylabel("Number of references")
    plt.tight_layout()
    plt.savefig(CHART_FILE, dpi=150)
    plt.close()
    print(f"\nChart saved to: {CHART_FILE}")

    return df
