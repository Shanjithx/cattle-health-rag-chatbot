"""
Comprehensive Local Test & Verification Suite for Supabase & PostgREST Safety
-------------------------------------------------------------------------------
Verifies:
1. URL sanitization and PostgREST PGRST125 elimination.
2. Canonical table normalization, alias resolution, and malicious path blocking.
3. ID sanitization and MD5 hashing validation across all 2,294 chunks in chunks.jsonl.
4. Dry-run and offline isolation guaranteeing zero database mutations.
"""

import os
import sys
import json
import hashlib
import re
from pathlib import Path

# Ensure offline execution for HuggingFace if offline
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DRY_RUN"] = "true"
os.environ["OFFLINE_MODE"] = "1"

import erp_tools
import migrate_to_supabase

def test_url_sanitization():
    print("\n--- Test 1: URL Sanitization & PGRST125 Prevention ---")
    test_cases = [
        ("https://xyz.supabase.co/rest/v1/", "https://xyz.supabase.co"),
        ("https://xyz.supabase.co/rest/v1", "https://xyz.supabase.co"),
        ("https://xyz.supabase.co/", "https://xyz.supabase.co"),
        ("https://xyz.supabase.co", "https://xyz.supabase.co"),
        ("  https://xyz.supabase.co/rest/v1///  ", "https://xyz.supabase.co"),
        ("http://localhost:54321/rest/v1/", "http://localhost:54321"),
        ("", ""),
        (None, "")
    ]
    
    for raw, expected in test_cases:
        res1 = erp_tools.sanitize_supabase_url(raw)
        res2 = migrate_to_supabase.sanitize_supabase_url(raw)
        assert res1 == expected, f"[erp_tools] Expected '{expected}', got '{res1}' for input '{raw}'"
        assert res2 == expected, f"[migrate_to_supabase] Expected '{expected}', got '{res2}' for input '{raw}'"
    
    # Test client PostgREST base_url resolution
    clean_url = "https://fgdinigajnupfmnrxzro.supabase.co"
    from supabase import create_client
    c = create_client(clean_url, "fake-key")
    base_url = str(c.postgrest.base_url)
    assert not base_url.endswith("/rest/v1/rest/v1"), f"PGRST125 Hazard: Malformed base_url {base_url}"
    assert base_url.endswith("/rest/v1"), f"Expected base_url to end with /rest/v1, got {base_url}"
    print("  [PASS] Supabase URL sanitization eliminates duplicate '/rest/v1/rest/v1' paths.")

def test_postgrest_table_routes():
    print("\n--- Test 2: Schema Table Normalization & Routing Security ---")
    valid_tables = [
        "farms", "users", "cattles", "employees", "employee_attendance",
        "cctv_cameras", "cctv_alerts", "rank_details", "inventory_items",
        "income_records", "milk_records", "cattle_mastitis_tests", "bcs_tests"
    ]
    
    from supabase import create_client
    client = create_client("https://fgdinigajnupfmnrxzro.supabase.co", "fake-key")
    
    for tbl in valid_tables:
        norm = erp_tools.normalize_table_name(tbl)
        assert norm == tbl, f"Normalization mismatch for valid table {tbl}: got {norm}"
        req_path = str(client.table(norm).select("*").request.path)
        assert "/rest/v1/rest/v1" not in req_path, f"PGRST125 bug in route: {req_path}"
        assert req_path.endswith(f"/rest/v1/{tbl}"), f"Unexpected path {req_path}"
    print(f"  [PASS] All {len(valid_tables)} core schema tables produce verified PostgREST routes.")

    # Test alias mapping
    alias_tests = [
        ("cattle", "cattles"),
        ("cow", "cattles"),
        ("cows", "cattles"),
        ("milk", "milk_records"),
        ("milking", "milk_records"),
        ("feed", "inventory_items"),
        ("inventory", "inventory_items"),
        ("alerts", "cctv_alerts"),
        ("attendance", "employee_attendance"),
        ("mastitis", "cattle_mastitis_tests"),
        ("bcs", "bcs_tests"),
        ("score", "rank_details"),
    ]
    for raw, expected in alias_tests:
        norm = erp_tools.normalize_table_name(raw)
        assert norm == expected, f"Alias failed: {raw} -> {norm} (expected {expected})"
    print(f"  [PASS] All {len(alias_tests)} natural language table aliases resolved correctly.")

    # Test dangerous / invalid table rejection (prevents PGRST125)
    invalid_cases = [
        "cattles/1",
        "cattles/status",
        "../cattles",
        "cattles; DROP TABLE users;--",
        "unknown_table_name",
        "",
        None,
        1234
    ]
    for inv in invalid_cases:
        norm = erp_tools.normalize_table_name(inv)
        assert norm is None, f"Dangerous input was not blocked: {inv} -> {norm}"
    print(f"  [PASS] All {len(invalid_cases)} malformed/dangerous table paths safely rejected.")

def test_chunk_id_sanitization_and_md5():
    print("\n--- Test 3: Local Chunk ID Sanitization & MD5 Validation (chunks.jsonl) ---")
    chunks_file = Path("chunks.jsonl")
    assert chunks_file.exists(), "chunks.jsonl not found in root directory!"
    
    stats = migrate_to_supabase.validate_chunks_dataset(chunks_file)
    
    assert stats["total_chunks"] == 2294, f"Expected 2294 chunks, got {stats['total_chunks']}"
    assert stats["unique_raw_ids"] == 2294, "Raw IDs have duplicate entries!"
    assert stats["unique_md5_hashes"] == 2294, "MD5 collision detected!"
    assert stats["collision_free"] is True, "Collision free check failed!"
    assert stats["all_hashes_32_chars_hex"] is True, "Not all hashes are 32-char hex strings!"
    assert stats["raw_ids_with_slashes"] == 2294, "Expected all raw IDs to contain slashes"
    assert stats["raw_ids_with_spaces"] > 2000, "Expected most raw IDs to contain spaces"
    
    # Test sample chunk payload
    sample_item = {
        "id": "bcs/BCS score.pdf_p1_c1",
        "text": "Condition scoring of dairy cows",
        "metadata": {"filename": "BCS score.pdf", "category": "bcs", "page": 1}
    }
    payload = migrate_to_supabase.prepare_chunk_payload(sample_item, [0.1] * 384)
    assert payload["chunk_id"] == "c5faffa85c5fe43bd05e2f3acd897c4d"
    assert "/" not in payload["chunk_id"]
    assert " " not in payload["chunk_id"]
    assert len(payload["chunk_id"]) == 32
    assert payload["content"] == sample_item["text"]
    assert payload["category"] == "bcs"
    
    print(f"  [PASS] Verified {stats['total_chunks']} chunks in chunks.jsonl.")
    print(f"  [PASS] 100% of raw IDs ({stats['raw_ids_with_slashes']} slashes, {stats['raw_ids_with_spaces']} spaces) sanitized to URL-safe MD5 hex strings.")
    print(f"  [PASS] Zero hash collisions found ({stats['unique_md5_hashes']}/{stats['total_chunks']} unique).")

def test_dry_run_migration():
    print("\n--- Test 4: Dry-Run Vector Migration (Zero Remote Mutations) ---")
    result = migrate_to_supabase.migrate(
        dry_run=True,
        batch_size=500,
        mock_embeddings=True
    )
    assert result["status"] == "success"
    assert result["dry_run"] is True
    assert result["total_chunks_processed"] == 2294
    assert result["total_batches"] == 5
    assert "/rest/v1/knowledge_chunks" in result["url_check"]
    print("  [PASS] Full dataset simulated batch migration completed in dry-run mode.")

def test_dry_run_erp_tools():
    print("\n--- Test 5: Dry-Run ERP Tools Execution (Zero Remote Mutations) ---")
    questions = [
        ("How much milk did I produce today?", "milk_records"),
        ("Show me my feed inventory.", "inventory_items"),
        ("List all registered cows.", "cattles"),
        ("What are the latest CCTV alerts?", "cctv_alerts"),
        ("What is my farm ranking and pashu points?", "rank_details"),
        ("Show employee attendance records.", "employee_attendance"),
    ]
    
    for q, expected_table in questions:
        res = erp_tools.execute_supabase_tool(q, farm_id=1, dry_run=True)
        assert res and len(res) > 10, f"Empty response for query '{q}'"
        assert expected_table in res or "records" in res.lower(), f"Expected table {expected_table} context in result: {res}"
        print(f"  [PASS] Query: '{q}' -> Handled cleanly via dry-run mock isolation.")

def test_mutation_guardrail():
    print("\n--- Test 6: Strict Database Mutation Guardrail ---")
    # Verify that attempting live mutation without ALLOW_LIVE_MUTATIONS raises PermissionError
    os.environ["ALLOW_LIVE_MUTATIONS"] = "false"
    try:
        migrate_to_supabase.migrate(
            dry_run=False,
            limit=5,
            mock_embeddings=True
        )
        assert False, "Migration should have been blocked by mutation guardrail!"
    except PermissionError as e:
        print(f"  [PASS] Mutation guardrail successfully intercepted unauthorized write: {e}")

if __name__ == "__main__":
    print("=" * 65)
    print(" Starting Supabase & PostgREST Safety Verification Suite")
    print("=" * 65)
    
    test_url_sanitization()
    test_postgrest_table_routes()
    test_chunk_id_sanitization_and_md5()
    test_dry_run_migration()
    test_dry_run_erp_tools()
    test_mutation_guardrail()
    
    print("\n" + "=" * 65)
    print(" ALL 6 VERIFICATION TEST SUITES PASSED SUCCESSFULLY (100% OK)")
    print("=" * 65)
