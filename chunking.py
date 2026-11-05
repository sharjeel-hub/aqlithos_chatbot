#!/usr/bin/env python3
"""Create overlapping, source-aware text chunks from a JSONL knowledge base.

Example (PowerShell):
    python chunking.py --input aqlithos_documents_cleaned.jsonl --output data/chunks.jsonl

This script uses only Python's standard library. It does not call an LLM,
create embeddings, or modify the input file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records and report the exact line for malformed JSON."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} must contain a JSON object.")
            records.append(record)
    return records


def choose_boundary(text: str, start: int, hard_end: int, min_end: int) -> int:
    """Prefer a line, sentence, or word boundary near the target chunk end."""
    newline = text.rfind("\n", min_end, hard_end)
    sentence = text.rfind(". ", min_end, hard_end)
    question = text.rfind("? ", min_end, hard_end)
    exclamation = text.rfind("! ", min_end, hard_end)
    space = text.rfind(" ", min_end, hard_end)

    # Choose the rightmost reasonable boundary; keep sentence punctuation.
    candidates = [
        (newline, 0),
        (sentence + 1 if sentence >= 0 else -1, 1),
        (question + 1 if question >= 0 else -1, 1),
        (exclamation + 1 if exclamation >= 0 else -1, 1),
        (space, 0),
    ]
    valid = [position for position, _ in candidates if position > start]
    return max(valid) if valid else hard_end


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into chunks, preserving a small overlap between neighbors."""
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    text_length = len(text)
    min_boundary = int(chunk_size * 0.70)

    while start < text_length:
        hard_end = min(start + chunk_size, text_length)
        if hard_end == text_length:
            end = hard_end
        else:
            end = choose_boundary(
                text,
                start,
                hard_end,
                min(start + min_boundary, hard_end - 1),
            )

        # Guard against pathological cases where no useful boundary is found.
        if end <= start:
            end = hard_end

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        # Carry overlap into the next chunk. Move to a nearby whitespace/newline
        # boundary so the next chunk is less likely to begin mid-word.
        next_start = max(start + 1, end - overlap)
        search_limit = min(end, next_start + 100)
        newline = text.find("\n", next_start, search_limit)
        space = text.find(" ", next_start, search_limit)
        boundaries = [p for p in (newline, space) if p >= 0]
        if boundaries:
            next_start = min(boundaries) + 1

        # Ensure forward progress even when overlap is configured too large.
        start = max(start + 1, next_start)

    return chunks


ZERO_COUNTER_LABELS = {
    "projects delivered",
    "ai solutions",
    "enterprise clients",
    "industries served",
    "years of experience",
    "countries supported",
}

def remove_zero_counter_placeholders(content: str) -> tuple[str, list[str]]:
    """Remove known zero-valued animated website counter placeholders.

    These are presentation placeholders from count-up widgets, not verified
    company statistics. Only exact known labels in a counter sequence match.
    """
    lines = content.splitlines()
    kept: list[str] = []
    removed: list[str] = []
    i = 0
    while i < len(lines):
        current = lines[i].strip().lower()
        if current == "0" and i + 1 < len(lines):
            next_line = lines[i + 1].strip().lower()
            if next_line == "+" and i + 2 < len(lines):
                label = lines[i + 2].strip()
                if label.lower() in ZERO_COUNTER_LABELS:
                    removed.append(label)
                    i += 3
                    continue
            elif next_line in ZERO_COUNTER_LABELS:
                removed.append(lines[i + 1].strip())
                i += 2
                continue
        kept.append(lines[i])
        i += 1
    return "\n".join(kept), removed


def make_chunk_records(
    document: dict[str, Any], chunk_size: int, overlap: int
) -> tuple[list[dict[str, Any]], list[str]]:
    content = document.get("content", "")
    if not isinstance(content, str) or not content.strip():
        return [], []

    content, removed_metrics = remove_zero_counter_placeholders(content)
    chunks = split_text(content, chunk_size, overlap)
    source_url = str(document.get("source_url", ""))
    original_hash = str(document.get("content_hash", ""))
    document_id = original_hash or hashlib.sha256(
        source_url.encode("utf-8")
    ).hexdigest()

    output: list[dict[str, Any]] = []
    total = len(chunks)
    for index, chunk_text in enumerate(chunks, start=1):
        chunk_hash = hashlib.sha256(
            f"{document_id}:{index}:{chunk_text}".encode("utf-8")
        ).hexdigest()
        output.append(
            {
                "chunk_id": chunk_hash[:20],
                "document_id": document_id,
                "chunk_index": index,
                "chunk_count": total,
                "source_url": source_url,
                "title": str(document.get("title", "")),
                "source_type": str(document.get("source_type", "")),
                "description": str(document.get("description", "")),
                "language": str(document.get("language", "")),
                "content_hash": original_hash,
                "collected_at": str(document.get("collected_at", "")),
                "content": chunk_text,
            }
        )
    return output, removed_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a cleaned JSONL knowledge base into overlapping text chunks."
    )
    parser.add_argument(
        "--input",
        default="aqlithos_documents_cleaned.jsonl",
        help="Input JSONL file (default: aqlithos_documents_cleaned.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="data/chunks.jsonl",
        help="Output chunk JSONL file (default: data/chunks.jsonl)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1800,
        help="Maximum target chunk size in characters (default: 1800)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=250,
        help="Approximate overlap between chunks in characters (default: 250)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        logging.error("Input file not found: %s", input_path)
        return 2
    if args.chunk_size < 200:
        logging.error("--chunk-size must be at least 200 characters.")
        return 2
    if args.chunk_overlap < 0 or args.chunk_overlap >= args.chunk_size:
        logging.error("--chunk-overlap must be >= 0 and smaller than --chunk-size.")
        return 2
    if input_path.resolve() == output_path.resolve():
        logging.error("Input and output paths must be different.")
        return 2

    try:
        documents = read_jsonl(input_path)
        output_records: list[dict[str, Any]] = []
        documents_skipped = 0
        documents_with_removed_metrics: list[str] = []
        removed_metric_count = 0
        for document in documents:
            chunks, removed_metrics = make_chunk_records(
                document, args.chunk_size, args.chunk_overlap
            )
            if not chunks:
                documents_skipped += 1
                continue
            output_records.extend(chunks)
            removed_metric_count += len(removed_metrics)
            if removed_metrics:
                documents_with_removed_metrics.append(
                    str(document.get("source_url", ""))
                )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8", newline="\n") as file:
            for record in output_records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        temp_path.replace(output_path)

        sizes = [len(record["content"]) for record in output_records]
        stats = {
            "input_file": str(input_path),
            "output_file": str(output_path),
            "generated_at": utc_now(),
            "settings": {
                "chunk_size_characters": args.chunk_size,
                "chunk_overlap_characters": args.chunk_overlap,
            },
            "summary": {
                "documents_read": len(documents),
                "documents_skipped_empty": documents_skipped,
                "chunks_created": len(output_records),
                "zero_counter_placeholders_removed": removed_metric_count,
                "documents_with_zero_counter_placeholders_removed": documents_with_removed_metrics,
                "average_chunk_characters": round(sum(sizes) / len(sizes), 2) if sizes else 0,
                "minimum_chunk_characters": min(sizes) if sizes else 0,
                "maximum_chunk_characters": max(sizes) if sizes else 0,
            },
        }
        stats_path = output_path.parent / "chunk_stats.json"
        stats_path.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        logging.info("Read %s documents", len(documents))
        logging.info("Created %s chunks", len(output_records))
        logging.info("Saved chunks: %s", output_path)
        logging.info("Saved statistics: %s", stats_path)
        return 0

    except (OSError, ValueError) as exc:
        logging.error("Chunking failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
