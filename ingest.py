
import pymupdf
import re
import json
from pathlib import Path

DOCUMENTS_DIR = Path("documents")
OUTPUT_FILE = Path("chunks.jsonl")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def clean_text(text):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_chunks(text):
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        if end < len(text):
            boundary = text.rfind(" ", start, end)

            if boundary > start + CHUNK_SIZE * 0.6:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - CHUNK_OVERLAP, start + 1)

        while start < len(text) and text[start].isspace():
            start += 1

    return chunks


def process_pdf(pdf_path):
    print(f"\nProcessing: {pdf_path.name}")

    document_chunks = []

    relative_path = pdf_path.relative_to(DOCUMENTS_DIR)
    category = (
        relative_path.parts[0]
        if len(relative_path.parts) > 1
        else "uncategorized"
    )

    try:
        with pymupdf.open(pdf_path) as doc:

            page_count = len(doc)
            empty_pages = 0

            for page_number, page in enumerate(doc, start=1):

                text = clean_text(page.get_text("text"))

                if not text:
                    empty_pages += 1
                    continue

                chunks = split_into_chunks(text)

                for chunk_index, chunk in enumerate(chunks, start=1):

                    document_chunks.append({
                        "id": f"{relative_path.as_posix()}_p{page_number}_c{chunk_index}",
                        "text": chunk,
                        "metadata": {
                            "source": relative_path.as_posix(),
                            "filename": pdf_path.name,
                            "category": category,
                            "page": page_number,
                            "chunk_index": chunk_index
                        }
                    })

        print(f"  Pages: {page_count}")
        print(f"  Chunks: {len(document_chunks)}")

        if empty_pages:
            print(f"  Empty pages: {empty_pages}")

    except Exception as e:
        print(f"  ERROR: {e}")
        return []

    return document_chunks


def main():

    if not DOCUMENTS_DIR.exists():
        print("Documents folder not found!")
        return

    # Find PDFs only once, regardless of extension case
    pdf_files = sorted(
        path for path in DOCUMENTS_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )

    if not pdf_files:
        print("No PDF files found!")
        return

    print("=" * 50)
    print("CATTLE RAG - DOCUMENT INGESTION")
    print("=" * 50)

    print(f"PDF files found: {len(pdf_files)}")

    all_chunks = []

    for pdf_path in pdf_files:
        chunks = process_pdf(pdf_path)
        all_chunks.extend(chunks)

    # Save only successfully processed chunks
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        for chunk in all_chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print("\n" + "=" * 50)
    print("INGESTION COMPLETED")
    print("=" * 50)

    print(f"Total PDFs: {len(pdf_files)}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Output saved to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
