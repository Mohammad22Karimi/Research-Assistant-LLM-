# LLM-based Research Assistant

Second project of the **Introduction to Data Mining** course.
A local, fully offline research assistant that reads a set of research
papers (PDF), builds a vector knowledge base, answers questions with RAG,
and can call analysis functions (results extraction, comparison, survey
generation, ...) when the user asks for them.

Everything runs locally: **Llama 3.1 8B via Ollama** for generation and
**all-MiniLM-L6-v2** for embeddings. No cloud APIs, no API keys.

---

## Project architecture

```
project/
├── papers/                  # put your 4+ paper PDFs here
├── database/                # persistent ChromaDB storage (auto-created)
├── utils/
│   ├── pdf_processing.py    # Stage 1: PDF -> text -> sections (PyMuPDF + regex headings)
│   ├── chunking.py          # Stage 2: 500-word chunks with 50-word overlap
│   ├── embeddings.py        # Stage 3: all-MiniLM-L6-v2 sentence embeddings
│   ├── vector_db.py         # Stage 4: ChromaDB store + similarity search
│   └── llm.py               # wrapper around local Llama 3.1 8B (Ollama)
├── rag/
│   ├── pipeline.py          # Stage 5: RAG pipeline + Stage 7: fallback
│   └── router.py            # Stage 6: LLM decides which function to call
├── functions/
│   ├── results_extraction.py  # extract_experimental_results()   [required]
│   ├── comparison.py          # compare_results()                [required]
│   ├── survey.py              # generate_survey()                [required]
│   ├── research_gap.py        # find_research_gap()              [bonus]
│   ├── references.py          # compare_references()             [bonus]
│   └── limitations.py         # compare_limitations()            [bonus]
├── main.py                  # Stage 8: menu-based CLI
├── requirements.txt
└── README.md
```

Data flow when building the knowledge base (menu option 1):

```
PDF file --(PyMuPDF)--> full text --(heading regex)--> sections
        --(chunking)--> 500-word chunks --(MiniLM)--> embeddings
        --(ChromaDB)--> database/  (text + paper + section + vector)
```

Data flow when asking a question (menu option 2):

```
question --> LLM router (one keyword) --> either:
   ANSWER            -> RAG pipeline (below)
   EXTRACT_RESULTS   -> extract_experimental_results()
   COMPARE_RESULTS   -> compare_results()
   GENERATE_SURVEY   -> generate_survey()
   RESEARCH_GAP / COMPARE_REFERENCES / COMPARE_LIMITATIONS (bonus)

RAG pipeline:
question -> embedding -> ChromaDB top-5 search
         -> best similarity >= 0.35 ?
              yes -> build context -> Llama 3.1 8B -> answer from papers
              no  -> "The answer was not found in the selected papers."
                     + clearly separated general LLM answer   (Stage 7)
```

## Chunking strategy

Fixed-size chunks of **500 words** with a **50-word overlap**, applied
*per section* (never across section boundaries).

Justification:

* 500 words ≈ one or two paragraphs — big enough to contain a complete
  idea, small enough that 5 retrieved chunks + prompt fit comfortably in
  the context window of an 8B model.
* The 50-word overlap ensures a sentence cut by a chunk boundary appears
  complete in at least one of the two neighbouring chunks.
* Chunking per section means every chunk carries a meaningful
  `(paper, section)` label, which the analysis functions use to fetch,
  for example, only the *results* chunks of one paper.
* Semantic chunking was intentionally avoided: it adds models and
  complexity without a clear benefit here, and this project values
  explainability over sophistication.

## Embedding model

`all-MiniLM-L6-v2` from *sentence-transformers*: small (~80 MB), fast on
CPU, 384-dimensional vectors, and the de-facto standard baseline for
semantic search. The same model embeds both the chunks (once, at build
time) and every user question (at query time).

## Vector database

**ChromaDB** with a persistent client (folder `database/`) and **cosine**
distance. ChromaDB was chosen over FAISS because it stores the chunk
text, the metadata (`paper`, `section`) and the embedding together and
persists everything automatically — with FAISS we would have to maintain
an index-to-text mapping ourselves. Cosine distance means
`similarity = 1 − distance`, which makes the fallback threshold easy to
interpret.

## RAG workflow

1. The question is embedded with the same MiniLM model.
2. ChromaDB returns the top-5 most similar chunks.
3. The chunks are concatenated into a context block; each chunk is
   labelled with its source paper and section so the model can cite them.
4. Llama 3.1 8B is instructed to answer **only** from that context.

**Fallback (Stage 7):** if even the best chunk's cosine similarity is
below **0.35**, the system prints
`The answer was not found in the selected papers.` and then asks the LLM
for a general-knowledge answer, printed under a separate, clearly
labelled heading. The threshold was tuned manually: in-scope questions
typically score > 0.4, unrelated ones < 0.3.

## Function calling workflow

No agent framework is used. Function calling is implemented in
`rag/router.py` in three transparent steps:

1. The user's request and a menu of 7 actions (with one-line
   descriptions) are shown to Llama 3.1 8B.
2. The model must reply with exactly one keyword
   (e.g. `COMPARE_RESULTS`).
3. Plain Python in `main.py` dispatches the keyword to the matching
   function. If the reply is unexpected, the system safely defaults to a
   normal RAG answer.

The three required functions follow the same simple recipe —
*fetch the relevant chunks by metadata → ask the LLM → post-process*:

* `extract_experimental_results()` — reads results/experiments chunks of
  each paper, asks the LLM for a small JSON of metrics (Accuracy,
  Precision, Recall, F1, mAP, IoU, ...), parses it into a pandas
  DataFrame (rows = papers, columns = metrics).
* `compare_results()` — prints that DataFrame as a comparison table and
  saves a grouped matplotlib bar chart to `comparison_chart.png`.
* `generate_survey()` — summarizes each paper individually, then asks the
  LLM to write a five-part mini survey (problem introduction, per-paper
  summaries, comparison, evolution of methods, conclusion); saved to
  `mini_survey.md`.

Bonus functions: `find_research_gap()` (limitations, unsolved problems,
future work), `compare_references()` (references counted per publication
year — pure regex, no LLM needed — table + chart) and
`compare_limitations()` (weaknesses of each method compared).

## How to run

1. **Install Ollama** (https://ollama.com) and pull the model once:

   ```bash
   ollama pull llama3.1:8b
   ```

2. **Install the Python packages** (Python 3.10+ recommended):

   ```bash
   pip install -r requirements.txt
   ```

3. **Add papers:** copy at least 4 PDFs of comparable papers (same topic,
   ideally same benchmark) into the `papers/` folder.

4. **Start the assistant:**

   ```bash
   python main.py
   ```

5. In the menu, first run **option 1** (build the knowledge base — needed
   once, or whenever the PDFs change), then use the other options, e.g.:

   ```
   Ask your question:
   > Which paper achieved the best accuracy?
   ```

## Required packages

| Package               | Used for                                   |
|-----------------------|--------------------------------------------|
| pymupdf               | PDF text extraction (Stage 1)              |
| sentence-transformers | chunk/question embeddings (Stage 3)        |
| chromadb              | vector database (Stage 4)                  |
| pandas                | metric tables (Stage 6)                    |
| matplotlib            | comparison charts (Stage 6)                |
| ollama                | talking to the local Llama 3.1 8B (Stage 5)|

## Known limitations

* Section detection is heuristic; papers with unusual layouts fall back
  to a single `full text` section (the system still works, only the
  section labels are less precise).
* Metric extraction depends on the LLM reading numbers correctly; very
  dense result tables in PDFs can lose structure during text extraction.
* The similarity threshold (0.35) is a manually tuned constant, not a
  learned value.
