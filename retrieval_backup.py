
import json
import re
import chromadb

from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


# ==========================================
# 1. CONFIGURATION
# ==========================================

DB_PATH = "./vector_db"
COLLECTION_NAME = "cattle_knowledge"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNKS_FILE = Path("chunks.jsonl")

RRF_K = 60

CANDIDATE_K = 20


# ==========================================
# 2. LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)


# ==========================================
# 3. LOAD RERANKING MODEL
# ==========================================

print("Loading cross-encoder reranking model...")

reranker = CrossEncoder(RERANKER_NAME)

print("Reranking model loaded!")


# ==========================================
# 4. CONNECT TO CHROMADB
# ==========================================

print("Connecting to ChromaDB...")

client = chromadb.PersistentClient(path=DB_PATH)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print("ChromaDB connected!")


# ==========================================
# 5. LOAD DOCUMENT CHUNKS
# ==========================================

print("Loading document chunks...")

if not CHUNKS_FILE.exists():
    raise FileNotFoundError("chunks.jsonl not found!")

chunks = []

with open(CHUNKS_FILE, "r", encoding="utf-8") as file:

    for line in file:

        if line.strip():
            chunks.append(json.loads(line))

if not chunks:
    raise ValueError("No document chunks found!")

print(f"Loaded {len(chunks)} chunks!")


# ==========================================
# 6. TOKENIZATION WITH STOPWORD REMOVAL
# ==========================================

STOP_WORDS = {
    "what", "is", "the", "a", "an",
    "in", "on", "at", "of", "for",
    "to", "and", "or", "are", "was",
    "were", "be", "been", "being",
    "does", "do", "did", "how",
    "why", "when", "where", "which",
    "with", "from", "by", "about",
    "can", "could", "would", "should",
    "this", "that", "these", "those",
    "it", "its", "their", "they",
    "he", "she", "we", "you", "i"
}


def tokenize(text):

    words = re.findall(r"\b\w+\b", text.lower())

    return [
        word for word in words
        if word not in STOP_WORDS
    ]


# ==========================================
# 7. INITIALIZE BM25
# ==========================================

print("Building BM25 index...")

tokenized_corpus = [
    tokenize(item["text"]) for item in chunks
]

bm25 = BM25Okapi(tokenized_corpus)

print("BM25 index ready!")

print("Hybrid retrieval and reranking system initialized!")


# ==========================================
# 8. VECTOR SEARCH
# ==========================================

def vector_search(query, top_k=20):

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count())
    )

    retrieved_chunks = []

    if not results["documents"] or not results["documents"][0]:
        return retrieved_chunks

    for i in range(len(results["documents"][0])):

        retrieved_chunks.append({

            "id": results["ids"][0][i],

            "text": results["documents"][0][i],

            "metadata": results["metadatas"][0][i],

            "distance": results["distances"][0][i]

        })

    return retrieved_chunks


# ==========================================
# 9. BM25 KEYWORD SEARCH
# ==========================================

def bm25_search(query, top_k=20):

    tokenized_query = tokenize(query)

    if not tokenized_query:
        return []

    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    retrieved_chunks = []

    for index in ranked_indices:

        if scores[index] <= 0:
            continue

        item = chunks[index]

        retrieved_chunks.append({

            "id": item["id"],

            "text": item["text"],

            "metadata": item["metadata"],

            "bm25_score": float(scores[index])

        })

        if len(retrieved_chunks) >= top_k:
            break

    return retrieved_chunks


# ==========================================
# 10. RECIPROCAL RANK FUSION
# ==========================================

def reciprocal_rank_fusion(
    vector_results,
    bm25_results,
    top_k=5,
    candidate_k=CANDIDATE_K
):

    fused_results = {}

    # --------------------------------------
    # Process vector rankings
    # --------------------------------------

    for rank, result in enumerate(vector_results, start=1):

        chunk_id = result["id"]

        if chunk_id not in fused_results:

            fused_results[chunk_id] = {

                "id": chunk_id,

                "text": result["text"],

                "metadata": result["metadata"],

                "rrf_score": 0.0,

                "vector_rank": None,

                "bm25_rank": None

            }

        fused_results[chunk_id]["rrf_score"] += (
            1 / (RRF_K + rank)
        )

        fused_results[chunk_id]["vector_rank"] = rank

        fused_results[chunk_id]["distance"] = result["distance"]


    # --------------------------------------
    # Process BM25 rankings
    # --------------------------------------

    for rank, result in enumerate(bm25_results, start=1):

        chunk_id = result["id"]

        if chunk_id not in fused_results:

            fused_results[chunk_id] = {

                "id": chunk_id,

                "text": result["text"],

                "metadata": result["metadata"],

                "rrf_score": 0.0,

                "vector_rank": None,

                "bm25_rank": None

            }

        fused_results[chunk_id]["rrf_score"] += (
            1 / (RRF_K + rank)
        )

        fused_results[chunk_id]["bm25_rank"] = rank

        fused_results[chunk_id]["bm25_score"] = result["bm25_score"]


    # --------------------------------------
    # Rank by RRF score
    # --------------------------------------

    ranked_results = sorted(
        fused_results.values(),
        key=lambda item: item["rrf_score"],
        reverse=True
    )

    # Keep a larger candidate pool
    candidates = ranked_results[:candidate_k]

    return candidates


# ==========================================
# 11. CROSS-ENCODER RERANKING
# ==========================================

def rerank_results(question, candidates, top_k=5):

    print("\nReranking retrieved passages...")

    if not candidates:
        return []

    # Create question-passage pairs
    pairs = [
        (question, candidate["text"])
        for candidate in candidates
    ]

    # Calculate relevance scores
    scores = reranker.predict(
        pairs,
        batch_size=32,
        show_progress_bar=False
    )

    # Attach scores to corresponding chunks
    reranked_results = []

    for candidate, score in zip(candidates, scores):

        candidate = candidate.copy()

        candidate["rerank_score"] = float(score)

        reranked_results.append(candidate)

    # Sort by cross-encoder relevance score
    reranked_results.sort(
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    # Return final top K passages
    return reranked_results[:top_k]


# ==========================================
# 12. COMPLETE HYBRID RETRIEVAL PIPELINE
# ==========================================

def hybrid_search(question, top_k=5):

    print("\nPerforming dense retrieval...")

    vector_results = vector_search(
        question,
        top_k=20
    )

    print("\nPerforming BM25 retrieval...")

    sparse_results = bm25_search(
        question,
        top_k=20
    )

    print("\nCombining results using RRF...")

    candidates = reciprocal_rank_fusion(
        vector_results,
        sparse_results,
        top_k=top_k,
        candidate_k=CANDIDATE_K
    )

    print(
        f"RRF candidate pool: {len(candidates)} chunks"
    )

    final_results = rerank_results(
        question,
        candidates,
        top_k=top_k
    )

    return final_results


# ==========================================
# 13. INTERACTIVE TESTING
# ==========================================

if __name__ == "__main__":

    print("\n" + "=" * 60)

    print("       CATTLE HEALTH - HYBRID RETRIEVAL")

    print("=" * 60)

    question = input(
        "\nEnter your cattle-health question: "
    ).strip()

    if not question:

        print("Please enter a valid question.")

    else:

        # Run complete hybrid retrieval and reranking
        final_results = hybrid_search(
            question,
            top_k=5
        )

        print("\n" + "=" * 60)

        print("FINAL RERANKED RESULTS")

        print("=" * 60)

        for index, result in enumerate(final_results, start=1):

            print(f"\nResult {index}")

            print(
                "Source:",
                result["metadata"].get("filename", "Unknown")
            )

            print(
                "Category:",
                result["metadata"].get("category", "Unknown")
            )

            print(
                "Page:",
                result["metadata"].get("page", "Unknown")
            )

            print(
                "Reranking Score:",
                round(result["rerank_score"], 4)
            )

            print(
                "RRF Score:",
                round(result["rrf_score"], 6)
            )

            print(
                "Vector Rank:",
                result["vector_rank"]
            )

            print(
                "BM25 Rank:",
                result["bm25_rank"]
            )

            print(
                "Text:",
                result["text"][:600]
            )

            print("-" * 60)
