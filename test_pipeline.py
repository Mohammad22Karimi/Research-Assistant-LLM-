import os
from utils import chunking, embeddings, pdf_processing, vector_db


def run_test():
    print("⏳ 1. Connecting to ChromaDB...")
    collection = vector_db.get_collection()

    # Clear the database for a clean test
    vector_db.clear(collection)

    print("📖 2. Processing papers and extracting text...")
    papers_dir = "papers"  # Directory containing your papers
    all_records = []

    for filename in os.listdir(papers_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(papers_dir, filename)
            paper_name, sections = pdf_processing.process_pdf(pdf_path)
            chunks = chunking.chunk_sections(paper_name, sections)
            all_records.extend(chunks)
            print(
                f"   ✅ Paper '{paper_name}' processed successfully ({len(chunks)} chunks)."
            )

    print(
        f"\n🧠 3. Generating embeddings for {len(all_records)} chunks (this may take a few seconds)..."
    )
    texts = [r["text"] for r in all_records]
    vectors = embeddings.embed_texts(texts)

    print("💾 4. Saving data to the vector database...")
    vector_db.add_chunks(collection, all_records, vectors)
    print("🎉 Vector database has been successfully created and saved to disk!\n")

    # --- Semantic search test ---
    test_question = "What dataset is used for object detection evaluation?"
    print(f"❓ Test question: '{test_question}'")
    print("🔍 Searching across the four papers...\n")

    question_vector = embeddings.embed_text(test_question)
    results = vector_db.search(collection, question_vector, top_k=2)

    for i, hit in enumerate(results, 1):
        print(f"--- Result {i} (Similarity: {hit['similarity']:.2f}) ---")
        print(f"📌 Source: Paper [{hit['paper']}] - Section [{hit['section']}]")
        print(f"📝 Text: {hit['text'][:250]}...\n")


if __name__ == "__main__":
    run_test()