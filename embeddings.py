
#!/usr/bin/env python3
"""Generate OpenAI embeddings for Aqlithos text chunks."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("data/chunks.jsonl")
DEFAULT_OUTPUT = Path("data/embeddings.jsonl")
DEFAULT_STATS = Path("data/embedding_stats.json")
DEFAULT_MODEL = "text-embedding-3-small"

METADATA_FIELDS = (
    "document_id",
    "chunk_index",
    "chunk_count",
    "source_url",
    "title",
    "source_type",
    "description",
    "language",
    "content_hash",
    "collected_at",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate OpenAI embeddings from Aqlithos chunks."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--stats-output", default=str(DEFAULT_STATS))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dimensions", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args()


def load_chunks(
    input_path: Path,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], int]:

    if not input_path.is_file():
        raise FileNotFoundError(f"Input not found: {input_path}")

    chunks = []
    seen_ids = set()
    skipped_empty = 0

    with input_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):

            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number} must contain a JSON object."
                )

            chunk_id = str(record.get("chunk_id") or "").strip()
            content = str(
                record.get("content") or record.get("text") or ""
            ).strip()

            if not chunk_id:
                raise ValueError(
                    f"Missing chunk_id on line {line_number}"
                )

            if chunk_id in seen_ids:
                raise ValueError(
                    f"Duplicate chunk_id: {chunk_id}"
                )

            seen_ids.add(chunk_id)

            if not content:
                skipped_empty += 1
                logging.warning("Skipping empty chunk: %s", chunk_id)
                continue

            normalized = dict(record)
            normalized["chunk_id"] = chunk_id
            normalized["content"] = content
            chunks.append(normalized)

            if limit is not None and len(chunks) >= limit:
                break

    if not chunks:
        raise ValueError("No valid text chunks found.")

    return chunks, skipped_empty


def build_metadata(chunk: dict[str, Any]) -> dict[str, Any]:

    metadata = {}

    for field in METADATA_FIELDS:
        value = chunk.get(field)

        if value is None:
            continue

        if isinstance(value, (str, int, float, bool)):
            metadata[field] = value
        else:
            metadata[field] = str(value)

    return metadata


def write_embedding_record(
    output_file: Any,
    chunk: dict[str, Any],
    vector: list[float],
    model: str,
) -> None:

    record = {
        "id": chunk["chunk_id"],
        "embedding_model": model,
        "embedding_dimensions": len(vector),
        "embedding": vector,
        "document": chunk["content"],
        "metadata": build_metadata(chunk),
    }

    output_file.write(
        json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
        ) + "\n"
    )


def main() -> int:

    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not 1 <= args.batch_size <= 2048:
        logging.error("Batch size must be between 1 and 2048.")
        return 2

    if args.dimensions is not None and args.dimensions < 1:
        logging.error("Dimensions must be positive.")
        return 2

    if args.limit is not None and args.limit < 1:
        logging.error("Limit must be positive.")
        return 2

    input_path = Path(args.input)
    output_path = Path(args.output)
    stats_path = Path(args.stats_output)

    try:
        chunks, skipped_empty = load_chunks(
            input_path,
            args.limit,
        )
    except (OSError, ValueError) as exc:
        logging.error("Input validation failed: %s", exc)
        return 2

    logging.info(
        "Validated %s chunks; skipped %s empty chunks.",
        len(chunks),
        skipped_empty,
    )

    logging.info(
        "Model: %s | Batch size: %s",
        args.model,
        args.batch_size,
    )

    if args.validate_only:
        logging.info("Validation-only mode: no API request made.")
        return 0

    # Read .env explicitly from the same folder as this script.
    try:
        from dotenv import dotenv_values
        from openai import OpenAI
    except ImportError:
        logging.error(
            "Install dependencies: "
            "python -m pip install openai python-dotenv"
        )
        return 2

    env_path = Path(__file__).resolve().parent / ".env"

    env_values = dotenv_values(dotenv_path=env_path)

    api_key = (
        env_values.get("OPENAI_API_KEY") or ""
    ).strip()

    if not api_key:
        logging.error(
            "OPENAI_API_KEY is missing or empty in %s",
            env_path,
        )
        return 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    # Explicitly use the key read from this project's .env.
    client = OpenAI(
        api_key=api_key,
        max_retries=5,
        timeout=60.0,
    )

    started = time.monotonic()
    total_tokens = 0
    dimensions_found = None
    embedded_count = 0
    batch_count = 0

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as out:

            for start in range(0, len(chunks), args.batch_size):

                batch = chunks[
                    start:start + args.batch_size
                ]

                texts = [
                    chunk["content"] for chunk in batch
                ]

                request_args = {
                    "model": args.model,
                    "input": texts,
                    "encoding_format": "float",
                }

                if args.dimensions is not None:
                    request_args["dimensions"] = args.dimensions

                response = client.embeddings.create(
                    **request_args
                )

                response_items = sorted(
                    response.data,
                    key=lambda item: item.index,
                )

                if len(response_items) != len(batch):
                    raise RuntimeError(
                        "API vector count does not match input count."
                    )

                for chunk, item in zip(batch, response_items):

                    vector = [
                        float(value)
                        for value in item.embedding
                    ]

                    if not vector:
                        raise RuntimeError(
                            f"Empty vector for {chunk['chunk_id']}"
                        )

                    if dimensions_found is None:
                        dimensions_found = len(vector)

                    elif len(vector) != dimensions_found:
                        raise RuntimeError(
                            "Vector dimensions changed during processing."
                        )

                    write_embedding_record(
                        out,
                        chunk,
                        vector,
                        args.model,
                    )

                    embedded_count += 1

                out.flush()

                batch_count += 1

                total_tokens += int(
                    getattr(response.usage, "total_tokens", 0) or 0
                )

                logging.info(
                    "Embedded %s/%s chunks (batch %s/%s).",
                    embedded_count,
                    len(chunks),
                    batch_count,
                    (len(chunks) + args.batch_size - 1)
                    // args.batch_size,
                )

        # Replace final file only after all batches succeed.
        temporary_path.replace(output_path)

        elapsed = round(
            time.monotonic() - started,
            2,
        )

        stats = {
            "input_file": str(input_path),
            "output_file": str(output_path),
            "model": args.model,
            "dimensions": dimensions_found,
            "batch_size": args.batch_size,
            "batches_completed": batch_count,
            "chunks_read": len(chunks) + skipped_empty,
            "chunks_embedded": embedded_count,
            "empty_chunks_skipped": skipped_empty,
            "total_input_tokens_reported_by_api": total_tokens,
            "elapsed_seconds": elapsed,
        }

        stats_path.write_text(
            json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        logging.info(
            "Successfully wrote %s embeddings to %s",
            embedded_count,
            output_path,
        )

        logging.info(
            "Vector dimensions: %s | API tokens: %s",
            dimensions_found,
            total_tokens,
        )

        logging.info("Statistics saved to %s", stats_path)

        return 0

    except Exception as exc:
        logging.exception(
            "Embedding generation failed: %s",
            exc,
        )

        logging.error(
            "Incomplete temporary file, if present: %s",
            temporary_path,
        )

        return 1


if __name__ == "__main__":
    sys.exit(main())
