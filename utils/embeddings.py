"""
Stage 3 - Embeddings
====================
Turns text (chunks or user questions) into dense vectors using the
SentenceTransformer model "all-MiniLM-L6-v2".

Why this model?
    * Small (~80 MB) and fast on CPU - runs on any student laptop.
    * Produces 384-dimensional sentence embeddings that work very well
      for semantic similarity search.
    * It is the standard "first choice" model in the sentence-transformers
      documentation, so the choice is easy to justify.
"""

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

# The model is loaded lazily (on first use) and kept in this module-level
# variable so it is only loaded ONCE per program run - loading takes a few
# seconds, embedding afterwards is fast.
_model = None


def get_model():
    """
    Purpose : Load the SentenceTransformer model once and reuse it.
    Inputs  : none.
    Outputs : the loaded SentenceTransformer model object.
    """
    global _model
    if _model is None:
        print(f"[embeddings] loading model '{MODEL_NAME}' ...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts):
    """
    Purpose : Compute an embedding vector for every text in a list.
    Inputs  : texts (list[str]) - the texts to embed.
    Outputs : embeddings (list[list[float]]) - one 384-dim vector per text.
    """
    model = get_model()
    vectors = model.encode(texts, show_progress_bar=False)
    return [v.tolist() for v in vectors]  # plain lists -> ChromaDB friendly


def embed_text(text):
    """
    Purpose : Convenience wrapper to embed a single string (e.g. a question).
    Inputs  : text (str).
    Outputs : one embedding vector (list[float]).
    """
    return embed_texts([text])[0]
