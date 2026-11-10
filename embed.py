
import json
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
print("EMBED.PY STARTED")
# Configuration
CHUNKS_FILE = Path("chunks.jsonl")
DB_PATH = "./vector_db"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "cattle_knowledge"

BATCH_SIZE = 64


def main():

    # Check whether chunks.jsonl exists
    if not CHUNKS_FILE.exists():
        print("ERROR: chunks.jsonl not found!")
        return

    print("=" * 50)
    print("CATTLE RAG - EMBEDDING GENERATION")
    print("=" * 50)

    # Load extracted chunks
    chunks = []

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    if not chunks:
        print("No chunks found!")
        return

    print(f"Loaded chunks: {len(chunks)}")

    # Load embedding model
    print("\nLoading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Embedding model loaded!")

    # Initialize persistent ChromaDB
    print("\nConnecting to ChromaDB...")

    client = chromadb.PersistentClient(path=DB_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    print("ChromaDB connected!")

    # Generate embeddings and store in batches
    print("\nGenerating embeddings...")

    for start in range(0, len(chunks), BATCH_SIZE):

        batch = chunks[start:start + BATCH_SIZE]

        texts = [item["text"] for item in batch]
        ids = [item["id"] for item in batch]
        metadatas = [item["metadata"] for item in batch]

        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings.tolist()
        )

        print(
            f"Processed {min(start + BATCH_SIZE, len(chunks))}"
            f" / {len(chunks)} chunks"
        )

    # Display results
    print("\n" + "=" * 50)
    print("EMBEDDING GENERATION COMPLETED")
    print("=" * 50)

    print(f"Total chunks: {len(chunks)}")
    print(f"Stored vectors: {collection.count()}")
    print(f"Database location: {DB_PATH}")
    print(f"Collection: {COLLECTION_NAME}")


if __name__ == "__main__":
    main()
