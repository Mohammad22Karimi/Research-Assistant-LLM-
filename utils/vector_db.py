"""
Stage 4 - Vector Database
=========================
Stores every chunk (text + paper name + section name + embedding) in a
persistent ChromaDB collection and provides similarity search over it.

Why ChromaDB (and not FAISS)?
    * One library does everything: it stores the text, the metadata AND the
      vector together, and it persists to disk automatically.
    * With FAISS we would need to maintain a separate mapping from vector
      index -> chunk text / metadata ourselves.  ChromaDB removes that
      bookkeeping, so the code stays shorter and simpler.
    * The collection is created with cosine distance, so
      similarity = 1 - distance, which makes the fallback threshold in
      Stage 7 easy to reason about.
"""

import os

import chromadb

# The database lives in the "database/" folder next to main.py, so the
# knowledge base survives between program runs (build once, query often).
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database")
COLLECTION_NAME = "papers"


def get_collection():
    """
    Purpose : Open (or create) the persistent ChromaDB collection.
    Inputs  : none.
    Outputs : a chromadb Collection object configured for cosine distance.
    """
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # distance = 1 - cosine similarity
    )
    return collection


def add_chunks(collection, records, embeddings):
    """
    Purpose : Store chunks and their embeddings in the database.
    Inputs  : collection - the ChromaDB collection (from get_collection()).
              records (list[dict])  - chunk dicts from chunking.chunk_sections().
              embeddings (list[list[float]]) - one vector per record,
                                               same order as records.
    Outputs : none (data is written to disk by ChromaDB).
    """
    collection.add(
        ids=[r["id"] for r in records],
        documents=[r["text"] for r in records],
        embeddings=embeddings,
        metadatas=[{"paper": r["paper"], "section": r["section"]} for r in records],
    )


def search(collection, query_embedding, top_k=5):
    """
    Purpose : Find the chunks most similar to a query embedding.
    Inputs  : collection      - the ChromaDB collection.
              query_embedding (list[float]) - embedding of the user question.
              top_k (int)     - how many chunks to return.
    Outputs : hits (list[dict]) - each with keys:
                  "text"       : chunk text
                  "paper"      : source paper
                  "section"    : source section
                  "similarity" : cosine similarity in [0, 1] (higher = closer)
    """
    result = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    hits = []
    # ChromaDB returns parallel lists wrapped in an outer list (one per query).
    for text, meta, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append(
            {
                "text": text,
                "paper": meta["paper"],
                "section": meta["section"],
                "similarity": 1.0 - distance,  # cosine distance -> similarity
            }
        )
    return hits


def get_chunks(collection, paper=None, sections=None):
    """
    Purpose : Fetch chunks by metadata WITHOUT similarity search.  Used by
              the analysis functions, e.g. "all 'results' chunks of paper X".
    Inputs  : collection      - the ChromaDB collection.
              paper (str)     - only chunks of this paper (None = any paper).
              sections (list[str]) - only chunks of these sections (None = all).
    Outputs : texts (list[str]) - the matching chunk texts.
    """
    # Build a ChromaDB "where" filter from the given criteria.
    conditions = []
    if paper is not None:
        conditions.append({"paper": paper})
    if sections is not None:
        conditions.append({"section": {"$in": sections}})

    if len(conditions) > 1:
        where = {"$and": conditions}
    elif len(conditions) == 1:
        where = conditions[0]
    else:
        where = None

    result = collection.get(where=where)
    return result["documents"]


def list_papers(collection):
    """
    Purpose : Return the names of all papers stored in the database.
    Inputs  : collection - the ChromaDB collection.
    Outputs : papers (list[str]) - sorted unique paper names.
    """
    result = collection.get()  # fetch all metadata
    papers = {meta["paper"] for meta in result["metadatas"]}
    return sorted(papers)


def clear(collection):
    """
    Purpose : Delete all stored chunks (used before re-building the
              knowledge base so old data does not mix with new data).
    Inputs  : collection - the ChromaDB collection.
    Outputs : none.
    """
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
