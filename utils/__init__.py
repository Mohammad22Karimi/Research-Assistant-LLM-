# utils package
# Contains the low-level building blocks of the system:
#   pdf_processing.py -> Stage 1: extract text + sections from PDFs
#   chunking.py       -> Stage 2: split text into overlapping chunks
#   embeddings.py     -> Stage 3: turn chunks into embedding vectors
#   vector_db.py      -> Stage 4: store / search chunks in ChromaDB
#   llm.py            -> small wrapper around the local Llama 3.1 8B model
