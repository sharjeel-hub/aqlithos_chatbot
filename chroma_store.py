#!/usr/bin/env python3
"""Import precomputed Aqlithos embeddings into a persistent ChromaDB collection.

This script does NOT call OpenAI and does NOT regenerate embeddings.
It reads the JSONL format written by embeddings.py:
  {id, embedding_model, embedding_dimensions, embedding, document, metadata}

Safe to rerun: collection.upsert() inserts new IDs and updates existing IDs.
It deliberately does not delete records that are absent from the input file;
that deletion/reconciliation belongs in the later incremental-sync pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_INPUT = Path("data/embeddings.jsonl")
DEFAULT_PERSIST_DIR = Path("chroma_db")
DEFAULT_STATS = Path("data/chroma_import_stats.json")
DEFAULT_COLLECTION = "aqlithos_knowledge"
DEFAULT_PROJECT_ID = "aqlithos"
DEFAULT_DIMENSIONS = 1536


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import existing Aqlithos vectors into persistent ChromaDB."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Embeddings JSONL file")
    parser.add_argument("--persist-dir", default=str(DEFAULT_PERSIST_DIR), help="Local Chroma persistence directory")
    parser.add_argument("--stats-output", default=str(DEFAULT_STATS), help="Import statistics JSON")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection name")
    parser.add_argument("--project-id", default=DEFAULT_PROJECT_ID, help="Project namespace stored in metadata")
    parser.add_argument("--batch-size", type=int, default=100, help="Records per Chroma upsert call")
    parser.add_argument("--expected-dimensions", type=int, default=DEFAULT_DIMENSIONS, help="Expected vector size")
    parser.add_argument("--validate-only", action="store_true", help="Validate JSONL without importing into ChromaDB")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def _safe_metadata(value: Any) -> str | int | float | bool | None:
    """Return a Chroma-compatible scalar, or None for unsupported values."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value
    return str(value)


def load_embedding_records(
    input_path: Path,
    expected_dimensions: int,
    project_id: str,
) -> tuple[list[dict[str, Any]], str, int]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Embeddings file not found: {input_path}")

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    model_seen: str | None = None
    dimensions_seen: int | None = None

    with input_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} must contain a JSON object")

            record_id = str(record.get("id") or "").strip()
            document = str(record.get("document") or "").strip()
            vector = record.get("embedding")
            model = str(record.get("embedding_model") or "").strip()

            if not record_id:
                raise ValueError(f"Missing 'id' on line {line_number}")
            if record_id in seen_ids:
                raise ValueError(f"Duplicate vector ID on line {line_number}: {record_id}")
            seen_ids.add(record_id)

            if not document:
                raise ValueError(f"Empty document text on line {line_number} ({record_id})")
            if not model:
                raise ValueError(f"Missing embedding_model on line {line_number} ({record_id})")
            if not isinstance(vector, list) or not vector:
                raise ValueError(f"Missing or invalid embedding vector on line {line_number} ({record_id})")

            if model_seen is None:
                model_seen = model
            elif model != model_seen:
                raise ValueError(f"Mixed embedding models found: {model_seen!r} and {model!r}")

            if dimensions_seen is None:
                dimensions_seen = len(vector)
            elif len(vector) != dimensions_seen:
                raise ValueError(f"Inconsistent vector dimensions on line {line_number}")

            if len(vector) != expected_dimensions:
                raise ValueError(
                    f"Expected {expected_dimensions} dimensions, found {len(vector)} "
                    f"on line {line_number} ({record_id})"
                )

            clean_vector: list[float] = []
            for index, number in enumerate(vector):
                if isinstance(number, bool) or not isinstance(number, (int, float)):
                    raise ValueError(f"Non-numeric vector value at line {line_number}, index {index}")
                number_float = float(number)
                if not math.isfinite(number_float):
                    raise ValueError(f"Non-finite vector value at line {line_number}, index {index}")
                clean_vector.append(number_float)

            raw_metadata = record.get("metadata") or {}
            if not isinstance(raw_metadata, dict):
                raise ValueError(f"Metadata must be a JSON object on line {line_number}")

            metadata: dict[str, str | int | float | bool] = {}
            for key, value in raw_metadata.items():
                if not isinstance(key, str) or not key.strip():
                    continue
                safe_value = _safe_metadata(value)
                if safe_value is not None:
                    metadata[key] = safe_value

            # Project-level namespace: enables filtering and future multi-project reuse.
            record_project = metadata.get("project_id")
            if record_project and record_project != project_id:
                raise ValueError(
                    f"Record {record_id} belongs to project {record_project!r}, "
                    f"not {project_id!r}"
                )
            metadata["project_id"] = project_id
            metadata["embedding_model"] = model
            metadata["embedding_dimensions"] = len(clean_vector)

            records.append(
                {
                    "id": record_id,
                    "document": document,
                    "embedding": clean_vector,
                    "metadata": metadata,
                    "embedding_model": model,
                }
            )

    if not records:
        raise ValueError("No embedding records found in the input file")

    return records, model_seen or "", dimensions_seen or 0


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.batch_size < 1:
        logging.error("--batch-size must be at least 1")
        return 2
    if args.expected_dimensions < 1:
        logging.error("--expected-dimensions must be at least 1")
        return 2
    if not args.collection or len(args.collection) > 63:
        logging.error("--collection must be a non-empty name of at most 63 characters")
        return 2
    if not args.project_id.strip():
        logging.error("--project-id cannot be empty")
        return 2

    input_path = Path(args.input)
    persist_dir = Path(args.persist_dir)
    stats_path = Path(args.stats_output)

    try:
        records, model, dimensions = load_embedding_records(
            input_path=input_path,
            expected_dimensions=args.expected_dimensions,
            project_id=args.project_id,
        )
    except (OSError, ValueError) as exc:
        logging.error("Input validation failed: %s", exc)
        return 2

    logging.info("Validated %s vectors.", len(records))
    logging.info("Embedding model: %s | Dimensions: %s", model, dimensions)
    logging.info("Project: %s | Collection: %s", args.project_id, args.collection)

    if args.validate_only:
        logging.info("Validation-only mode: no ChromaDB writes were made.")
        return 0

    try:
        import chromadb
    except ImportError:
        logging.error("ChromaDB is not installed. Run: python -m pip install chromadb")
        return 2

    persist_dir.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    try:
        client = chromadb.PersistentClient(path=str(persist_dir.resolve()))
        collection = client.get_or_create_collection(
            name=args.collection,
            metadata={
                "project_id": args.project_id,
                "embedding_model": model,
                "embedding_dimensions": dimensions,
                "description": "Aqlithos company knowledge base",
            },
            configuration={"hnsw": {"space": "cosine"}},
        )

        # Refuse accidental cross-project reuse of an existing collection.
        existing_metadata = collection.metadata or {}
        existing_project = existing_metadata.get("project_id")
        if existing_project and existing_project != args.project_id:
            raise RuntimeError(
                f"Collection {args.collection!r} belongs to project {existing_project!r}, "
                f"not {args.project_id!r}. Choose a different collection name."
            )

        existing_model = existing_metadata.get("embedding_model")
        if existing_model and existing_model != model:
            raise RuntimeError(
                f"Collection uses embedding model {existing_model!r}, but input uses {model!r}. "
                "Do not mix embedding models in one collection."
            )

        existing_dimensions = existing_metadata.get("embedding_dimensions")
        if existing_dimensions and int(existing_dimensions) != dimensions:
            raise RuntimeError(
                f"Collection dimension is {existing_dimensions}, but input dimension is {dimensions}. "
                "Use a separate collection for a different vector dimension."
            )

        # Ensure identifying metadata exists even if the collection pre-existed.
        merged_metadata = dict(existing_metadata)
        merged_metadata.update(
            {
                "project_id": args.project_id,
                "embedding_model": model,
                "embedding_dimensions": dimensions,
                "description": "Aqlithos company knowledge base",
            }
        )
        if merged_metadata != existing_metadata:
            collection.modify(metadata=merged_metadata)

        total = len(records)
        batches = (total + args.batch_size - 1) // args.batch_size

        for start in range(0, total, args.batch_size):
            batch = records[start : start + args.batch_size]
            collection.upsert(
                ids=[item["id"] for item in batch],
                embeddings=[item["embedding"] for item in batch],
                documents=[item["document"] for item in batch],
                metadatas=[item["metadata"] for item in batch],
            )
            current_batch = start // args.batch_size + 1
            logging.info(
                "Upserted %s/%s vectors (batch %s/%s).",
                min(start + len(batch), total),
                total,
                current_batch,
                batches,
            )

        # Verify every input ID can be read back from this collection.
        input_ids = [item["id"] for item in records]
        returned_ids: list[str] = []
        for start in range(0, len(input_ids), args.batch_size):
            id_batch = input_ids[start : start + args.batch_size]
            result = collection.get(ids=id_batch, include=["metadatas"])
            returned_ids.extend(result.get("ids", []))

        missing_ids = sorted(set(input_ids) - set(returned_ids))
        if missing_ids:
            raise RuntimeError(
                f"ChromaDB verification failed: {len(missing_ids)} input IDs were not found. "
                f"First missing IDs: {missing_ids[:5]}"
            )

        elapsed = round(time.monotonic() - started, 2)
        stats = {
            "project_id": args.project_id,
            "collection": args.collection,
            "persist_directory": str(persist_dir.resolve()),
            "input_file": str(input_path),
            "embedding_model": model,
            "embedding_dimensions": dimensions,
            "distance_metric": "cosine",
            "vectors_upserted_from_input": total,
            "input_ids_verified": len(returned_ids),
            "collection_count_after_import": collection.count(),
            "batch_size": args.batch_size,
            "batches_completed": batches,
            "openai_api_calls_made": 0,
            "elapsed_seconds": elapsed,
        }
        stats_path.write_text(
            json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        logging.info("ChromaDB collection count: %s", collection.count())
        logging.info("Verified %s/%s input vector IDs.", len(returned_ids), total)
        logging.info("ChromaDB data persisted at: %s", persist_dir.resolve())
        logging.info("Import statistics saved to: %s", stats_path)
        logging.info("Import complete. No OpenAI API calls were made.")
        return 0

    except Exception as exc:
        logging.exception("ChromaDB import failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
