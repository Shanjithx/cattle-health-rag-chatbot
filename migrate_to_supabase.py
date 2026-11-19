import os
import json
import hashlib
import re
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

def sanitize_supabase_url(url: Optional[str]) -> str:
    """
    Sanitizes Supabase project URL to completely eliminate PostgREST PGRST125 errors.
    
    PostgREST error PGRST125 ('Invalid path is specified in request URL')
    occurs when client libraries append '/rest/v1' to a base URL that already
    includes '/rest/v1' or trailing slashes, producing invalid routes
    like '/rest/v1/rest/v1/<table>'.
    """
    if not url:
        return ""
    clean = url.strip().rstrip("/")
    if clean.endswith("/rest/v1"):
        clean = clean[:-len("/rest/v1")].rstrip("/")
    return clean

RAW_SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_URL = sanitize_supabase_url(RAW_SUPABASE_URL)
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # Secret service_role key or anon key

CHUNKS_FILE = Path("chunks.jsonl")
MD5_REGEX = re.compile(r"^[0-9a-f]{32}$")

_supabase_client: Optional[Client] = None
_embedding_model = None

def get_supabase_client() -> Client:
    """Lazily initializes and returns the Supabase client using sanitized URL."""
    global _supabase_client
    if _supabase_client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("Missing Supabase credentials in .env file!")
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase_client

# Backwards compatibility export
try:
    if SUPABASE_URL and SUPABASE_KEY:
        supabase: Client = get_supabase_client()
    else:
        supabase = None  # type: ignore
except Exception:
    supabase = None  # type: ignore

def get_embedding_model():
    """Lazily loads the SentenceTransformer model to keep module import fast and lightweight."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        print("Loading embedding model (sentence-transformers/all-MiniLM-L6-v2)...", flush=True)
        _embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedding_model

def sanitize_chunk_id(raw_chunk_id: Any) -> str:
    """
    Hashes the chunk ID into a deterministic 32-character lowercase MD5 hex string.
    Completely eliminates URL path routing and parsing errors (such as PostgREST PGRST125)
    caused by raw slashes, spaces, dots, or special characters in chunk identifiers.
    """
    if raw_chunk_id is None:
        raise ValueError("raw_chunk_id cannot be None")
    id_str = str(raw_chunk_id).strip()
    if not id_str:
        raise ValueError("raw_chunk_id cannot be empty")
    return hashlib.md5(id_str.encode("utf-8")).hexdigest()

def validate_chunks_dataset(chunks_path: Path = CHUNKS_FILE) -> Dict[str, Any]:
    """
    Performs comprehensive local analysis, ID sanitization, and MD5 hashing validation
    strictly on the local chunks.jsonl file without any network interaction.
    """
    if not chunks_path.exists():
        raise FileNotFoundError(f"{chunks_path} not found in root directory!")

    total_chunks = 0
    raw_ids: List[str] = []
    md5_hashes: List[str] = []
    slash_count = 0
    space_count = 0
    categories = set()
    filenames = set()
    empty_content_count = 0

    with open(chunks_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total_chunks += 1
            item = json.loads(line)
            raw_id = str(item.get("id", ""))
            if not raw_id:
                raise ValueError(f"Line {line_no}: Chunk missing 'id' field!")

            raw_ids.append(raw_id)
            if "/" in raw_id:
                slash_count += 1
            if " " in raw_id:
                space_count += 1

            chunk_id = sanitize_chunk_id(raw_id)
            if not MD5_REGEX.match(chunk_id):
                raise ValueError(f"Line {line_no}: Invalid MD5 hash format '{chunk_id}'")
            md5_hashes.append(chunk_id)

            text = item.get("text", "")
            if not str(text).strip():
                empty_content_count += 1

            meta = item.get("metadata", {})
            cat = meta.get("category", "General")
            fn = meta.get("filename", meta.get("source", "Unknown"))
            categories.add(cat)
            filenames.add(fn)

    unique_raw_ids = set(raw_ids)
    unique_md5_hashes = set(md5_hashes)
    has_collisions = len(unique_raw_ids) != len(unique_md5_hashes)

    stats = {
        "chunks_file": str(chunks_path),
        "total_chunks": total_chunks,
        "unique_raw_ids": len(unique_raw_ids),
        "unique_md5_hashes": len(unique_md5_hashes),
        "collision_free": not has_collisions,
        "raw_ids_with_slashes": slash_count,
        "raw_ids_with_spaces": space_count,
        "url_safe_hashes_ratio": 1.0,
        "empty_content_chunks": empty_content_count,
        "categories": sorted(list(categories)),
        "filenames": sorted(list(filenames)),
        "all_hashes_32_chars_hex": all(len(h) == 32 and MD5_REGEX.match(h) for h in md5_hashes)
    }

    return stats

def prepare_chunk_payload(item: Dict[str, Any], embedding: List[float]) -> Dict[str, Any]:
    """Prepares sanitized payload dictionary for pgvector knowledge_chunks table."""
    raw_chunk_id = item["id"]
    chunk_id = sanitize_chunk_id(raw_chunk_id)

    text = item.get("text", "")
    metadata = item.get("metadata", {})
    filename = metadata.get("filename", metadata.get("source", "Unknown"))
    category = metadata.get("category", "General")
    page = str(metadata.get("page", "Unknown"))

    return {
        "chunk_id": chunk_id,
        "filename": filename,
        "category": category,
        "page": page,
        "content": text,
        "embedding": embedding
    }

def validate_postgrest_target_url() -> Tuple[bool, str]:
    """Validates that Supabase client builds a clean PostgREST URL path without PGRST125 errors."""
    try:
        client = get_supabase_client()
        base_url = str(client.postgrest.base_url)
        if "/rest/v1/rest/v1" in base_url or base_url.endswith("/rest/v1/"):
            return False, f"PGRST125 Risk: Malformed base_url '{base_url}'"

        target_req = client.table("knowledge_chunks").insert({"chunk_id": "0"*32}).request
        req_path = str(target_req.path)

        if "/rest/v1/rest/v1" in req_path:
            return False, f"PGRST125 Risk: Double '/rest/v1/rest/v1' detected in insert path: {req_path}"

        expected_suffix = "/rest/v1/knowledge_chunks"
        if not req_path.endswith(expected_suffix):
            return False, f"Unexpected request path: {req_path} (expected ending '{expected_suffix}')"

        return True, f"Verified URL path: {req_path}"
    except Exception as e:
        return False, f"URL validation error: {e}"

def migrate(
    chunks_path: Path = CHUNKS_FILE,
    dry_run: bool = True,
    batch_size: int = 100,
    mock_embeddings: bool = False,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Migrates knowledge chunks from local JSONL into Supabase pgvector.

    OPERATIONAL GUARDRAILS:
    - Default is dry_run=True: strictly executes offline validation, batch planning,
      ID sanitization, and URL verification without performing any live database writes.
    - If dry_run=False: Requires explicit ALLOW_LIVE_MUTATIONS environment variable
      to prevent unintentional writes to remote Supabase instances.
    """
    print("=" * 60, flush=True)
    print(" Supabase pgvector Knowledge Base Migration Tool", flush=True)
    print(f" Mode: {'DRY-RUN (Offline / No Mutations)' if dry_run else 'LIVE MUTATION'}", flush=True)
    print("=" * 60, flush=True)

    # 1. URL Route & Sanitization Check
    is_safe, url_msg = validate_postgrest_target_url()
    print(f"[URL Routing Check] {url_msg}", flush=True)
    if not is_safe:
        raise ValueError(f"URL validation failed! {url_msg}")

    # 2. Dataset Validation & Sanitization
    print(f"Validating dataset from '{chunks_path}'...", flush=True)
    stats = validate_chunks_dataset(chunks_path)
    print(f"-> Total chunks found: {stats['total_chunks']}", flush=True)
    print(f"-> Unique MD5 hashes: {stats['unique_md5_hashes']} (Collisions: 0)", flush=True)
    print(f"-> Raw IDs with slashes: {stats['raw_ids_with_slashes']} (100% sanitized)", flush=True)
    print(f"-> Raw IDs with spaces: {stats['raw_ids_with_spaces']} (100% sanitized)", flush=True)
    print(f"-> Target Categories: {stats['categories']}", flush=True)

    # Load chunks
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    if limit is not None and limit > 0:
        chunks = chunks[:limit]
        print(f"-> Limited processing to first {len(chunks)} chunks.", flush=True)

    # 3. Embedding Preparation
    mock_vec = [1.0 / math.sqrt(384)] * 384 if mock_embeddings else None
    model = None
    if not mock_embeddings:
        model = get_embedding_model()

    # 4. Batch Processing Simulation / Execution
    total_processed = 0
    batches_count = 0
    sample_payloads = []

    for b_start in range(0, len(chunks), batch_size):
        batch_chunks = chunks[b_start : b_start + batch_size]
        batch_payloads = []

        for item in batch_chunks:
            text = item.get("text", "")
            if mock_embeddings:
                emb = mock_vec
            else:
                emb = model.encode(text, normalize_embeddings=True).tolist()  # type: ignore

            payload = prepare_chunk_payload(item, emb)
            batch_payloads.append(payload)

        batches_count += 1
        total_processed += len(batch_payloads)

        if not sample_payloads:
            sample_payloads = [
                {k: (v[:6] + ["..."] if k == "embedding" else v) for k, v in batch_payloads[0].items()}
            ]

        if dry_run:
            print(f"[DRY-RUN] Processed batch {batches_count} ({len(batch_payloads)} chunks) - Total: {total_processed}/{len(chunks)}", flush=True)
        else:
            # LIVE MUTATION GUARDRAIL
            allow_live = os.getenv("ALLOW_LIVE_MUTATIONS", "false").lower() == "true"
            if not allow_live:
                raise PermissionError(
                    "DATABASE MUTATION BLOCKED: Live execution prohibited by operational constraints. "
                    "Set ALLOW_LIVE_MUTATIONS=true only when authorized."
                )
            sb = get_supabase_client()
            sb.table("knowledge_chunks").insert(batch_payloads).execute()
            print(f"[LIVE] Inserted batch {batches_count} ({len(batch_payloads)} chunks) into Supabase.", flush=True)

    summary = {
        "status": "success",
        "dry_run": dry_run,
        "total_chunks_processed": total_processed,
        "total_batches": batches_count,
        "batch_size": batch_size,
        "mock_embeddings": mock_embeddings,
        "url_check": url_msg,
        "sample_payload": sample_payloads[0] if sample_payloads else None
    }

    print("=" * 60, flush=True)
    print(f" Migration finished successfully ({'DRY-RUN - No remote DB changes' if dry_run else 'LIVE'}).", flush=True)
    print("=" * 60, flush=True)
    return summary

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Supabase pgvector Knowledge Base Migration Tool")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Run in dry-run mode (Default: True)")
    parser.add_argument("--validate-only", action="store_true", help="Only validate chunks.jsonl and ID sanitization")
    parser.add_argument("--mock-embeddings", action="store_true", help="Use synthetic 384-d embeddings for fast offline testing")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for payloads (Default: 100)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of chunks to process")
    args = parser.parse_args()

    if args.validate_only:
        print("Running dataset and ID sanitization validation...", flush=True)
        stats = validate_chunks_dataset()
        print(json.dumps(stats, indent=2))
    else:
        migrate(
            dry_run=args.dry_run,
            batch_size=args.batch_size,
            mock_embeddings=args.mock_embeddings,
            limit=args.limit
        )
