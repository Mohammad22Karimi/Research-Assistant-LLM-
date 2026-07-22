"""
Stage 5 - RAG pipeline  +  Stage 7 - Fallback
=============================================
Implements the required pipeline:

    User Question -> Embedding -> Similarity Search -> Top-k retrieval
    -> Context construction -> local Llama 3.1 8B -> Answer

and the required fallback behaviour: if the best retrieved chunk is not
similar enough to the question, the system says the answer was not found in
the papers and then gives a clearly separated general-knowledge answer.
"""

from utils import embeddings, llm, vector_db

TOP_K = 5  # how many chunks are retrieved for each question

# If even the BEST retrieved chunk has cosine similarity below this value,
# we assume the papers do not contain the answer.  0.35 was chosen by
# manually trying a few in-scope and out-of-scope questions: related
# questions usually score well above 0.4, unrelated ones well below 0.3.
SIMILARITY_THRESHOLD = 0.30


def retrieve(question, top_k=TOP_K):
    """
    Purpose : Embed the question and fetch the top-k most similar chunks.
    Inputs  : question (str) - the user's question.
              top_k (int)    - number of chunks to retrieve.
    Outputs : hits (list[dict]) - chunks with paper/section/similarity info,
              sorted by similarity (ChromaDB returns them best-first).
    """
    question_embedding = embeddings.embed_text(question)
    collection = vector_db.get_collection()
    return vector_db.search(collection, question_embedding, top_k=top_k)


def build_context(hits):
    """
    Purpose : Turn retrieved chunks into one context string for the LLM.
              Each chunk is labelled with its source paper and section, so
              the model can say WHERE information comes from.
    Inputs  : hits (list[dict]) - output of retrieve().
    Outputs : context (str) - formatted context block.
    """
    parts = []
    for i, hit in enumerate(hits, start=1):
        parts.append(
            f"[Chunk {i} | paper: {hit['paper']} | section: {hit['section']}]\n"
            f"{hit['text']}"
        )
    return "\n\n".join(parts)


def answer_from_papers(question, hits):
    """
    Purpose : Ask the local LLM to answer using ONLY the retrieved context.
    Inputs  : question (str), hits (list[dict]).
    Outputs : answer (str).
    """
    context = build_context(hits)
    prompt = (
        "Answer the question using ONLY the context below, which was "
        "retrieved from a set of research papers. Mention which paper the "
        "information comes from. If the context is not sufficient, say so.\n\n"
        f"=== CONTEXT ===\n{context}\n\n"
        f"=== QUESTION ===\n{question}"
    )
    return llm.ask_llm(
        prompt,
        system="You are a precise research assistant that answers questions about scientific papers.",
    )


def answer_general(question):
    """
    Purpose : Fallback - answer from the LLM's own general knowledge
              (used when the papers do not contain the answer).
    Inputs  : question (str).
    Outputs : answer (str).
    """
    return llm.ask_llm(
        question,
        system="You are a helpful assistant. Answer from your general knowledge.",
    )


def ask(question):
    """
    Purpose : Full Stage 5 + Stage 7 behaviour for one user question.
    Inputs  : question (str).
    Outputs : response (str) - either an answer grounded in the papers, or
              the required fallback message followed by a clearly separated
              general-knowledge answer.
    """
    hits = retrieve(question)

    # hits[0] is the most similar chunk; if even that one is weak,
    # the knowledge base does not cover this question.
    best_similarity = hits[0]["similarity"] if hits else 0.0
    print(f"[rag] best chunk similarity = {best_similarity:.2f} "
          f"(threshold = {SIMILARITY_THRESHOLD})")

    if best_similarity >= SIMILARITY_THRESHOLD:
        answer = answer_from_papers(question, hits)
        return "=== Answer from papers ===\n" + answer

    # ---- Stage 7 fallback ----
    general = answer_general(question)
    return (
        "The answer was not found in the selected papers.\n"
        "But using the language model, a general answer is:\n\n"
        "=== General LLM answer (NOT from the papers) ===\n" + general
    )
