"""
Stage 2 - Chunking
==================
Splits the text of each section into fixed-size overlapping word chunks.

Why this strategy (fixed size + overlap) was chosen:
    * CHUNK_SIZE = 500 words:
        - Large enough that a chunk usually contains a complete idea
          (a paragraph or two), so retrieval returns meaningful context.
        - Small enough that several chunks fit into the context window of
          Llama 3.1 8B together with the question and the prompt.
    * OVERLAP = 50 words:
        - A sentence that sits exactly on a chunk boundary would be cut in
          half.  Repeating the last 50 words of a chunk at the start of the
          next chunk guarantees every sentence appears complete in at
          least one chunk.
    * No semantic chunking:
        - Semantic chunking needs extra models and logic that are hard to
          justify line-by-line in a short presentation.  Fixed-size
          chunking is deterministic, easy to explain, and works well
          because we already split the paper by section first (so chunks
          never mix, e.g., Introduction text with Results text).
"""

CHUNK_SIZE = 500  # words per chunk
OVERLAP = 50      # words shared between consecutive chunks


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """
    Purpose : Split one piece of text into overlapping word chunks.
    Inputs  : text (str)        - the text to split.
              chunk_size (int)  - number of words per chunk.
              overlap (int)     - number of words repeated between chunks.
    Outputs : chunks (list[str]) - list of chunk strings (possibly length 1
              if the text is shorter than chunk_size).
    """
    words = text.split()  # simple whitespace tokenization
    if not words:
        return []

    chunks = []
    step = chunk_size - overlap  # how far the window moves each time
    start = 0
    while start < len(words):
        chunk_words = words[start : start + chunk_size]
        chunks.append(" ".join(chunk_words))
        # Stop when this chunk already reached the end of the text,
        # otherwise the overlap would generate a duplicate tail chunk.
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


def chunk_sections(paper_name, sections):
    """
    Purpose : Apply chunk_text() to every section of a paper and attach
              metadata (paper name + section name) to each chunk.
    Inputs  : paper_name (str) - identifier of the paper.
              sections (dict)  - section name -> section text (Stage 1 output).
    Outputs : records (list[dict]) - one dict per chunk with keys:
                  "id"      : unique string id (paper_section_index)
                  "text"    : the chunk text
                  "paper"   : source paper name
                  "section" : source section name
    """
    records = []
    for section_name, section_text in sections.items():
        for i, chunk in enumerate(chunk_text(section_text)):
            records.append(
                {
                    "id": f"{paper_name}_{section_name}_{i}".replace(" ", "-"),
                    "text": chunk,
                    "paper": paper_name,
                    "section": section_name,
                }
            )
    return records
