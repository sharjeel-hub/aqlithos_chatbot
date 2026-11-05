
import json
import math
from pathlib import Path

import chromadb

# Project configuration
INPUT_FILE = Path("data/embeddings.jsonl")
CHROMA_HOST = "127.0.0.1"
CHROMA_PORT = 8000

PROJECT_ID = "aqlithos"
COLLECTION_NAME = "aqlithos_knowledge"
BATCH_SIZE = 32
EXPECTED_DIMENSIONS = 1536


def load_and_validate():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    records = [
        json.loads(line)
        for line in INPUT_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if len(records) != 86:
        raise ValueError(f"Expected 86 records, found {len(records)}")

    ids = []

    for index, record in enumerate(records, start=1):
        embedding = record.get("embedding")
        metadata = record.get("metadata")
        document = record.get("document")

        if not isinstance(metadata, dict):
            raise ValueError(f"Record {index}: invalid metadata")

        if not isinstance(document, str) or not document.strip():
            raise ValueError(f"Record {index}: missing document")

        if not isinstance(embedding, list) or len(embedding) != EXPECTED_DIMENSIONS:
            raise ValueError(f"Record {index}: invalid embedding dimensions")

        for value in embedding:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"Record {index}: invalid embedding value")

        for value in metadata.values():
            if not isinstance(value, (str, int, float, bool)):
                raise ValueError(f"Record {index}: unsupported metadata value")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"Record {index}: non-finite metadata value")

        ids.append(record["id"])

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs detected")

    print(f"Validated {len(records)} records.")
    return records


def main():
    records = load_and_validate()

    print("Connecting to Docker ChromaDB...")

    client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
    )

    print("Connection successful.")

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        metadata={
            "hnsw:space": "cosine",
            "project_id": PROJECT_ID,
            "embedding_model": "text-embedding-3-small",
            "embedding_dimensions": EXPECTED_DIMENSIONS,
        },
    )

    print(f"Collection: {COLLECTION_NAME}")

    # Upsert makes the import safe to retry:
    # existing IDs are updated instead of duplicated.
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]

        ids = [record["id"] for record in batch]
        documents = [record["document"] for record in batch]
        embeddings = [record["embedding"] for record in batch]

        metadatas = []

        for record in batch:
            metadata = dict(record["metadata"])
            metadata["project_id"] = PROJECT_ID
            metadata["embedding_model"] = record["embedding_model"]
            metadata["embedding_dimensions"] = int(
                record["embedding_dimensions"]
            )
            metadatas.append(metadata)

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        print(
            f"Imported batch: "
            f"{start + 1}-{start + len(batch)}"
        )

    # Verify the final collection
    total = collection.count()

    if total != len(records):
        raise RuntimeError(
            f"Import verification failed: expected 86, found {total}"
        )

    sample = collection.get(
        ids=[records[0]["id"]],
        include=["documents", "metadatas"],
    )

    if not sample["ids"]:
        raise RuntimeError("Sample record verification failed")

    result = collection.query(
        query_embeddings=[records[0]["embedding"]],
        n_results=1,
        include=["distances"],
    )

    print("\n========== IMPORT SUCCESS ==========")
    print(f"Project ID: {PROJECT_ID}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Total vectors: {total}")
    print(f"Dimensions: {EXPECTED_DIMENSIONS}")
    print(f"Embedding model: text-embedding-3-small")
    print(f"Sample retrieved ID: {result['ids'][0][0]}")
    print(f"Sample distance: {result['distances'][0][0]}")
    print("====================================")


if __name__ == "__main__":
    main()
